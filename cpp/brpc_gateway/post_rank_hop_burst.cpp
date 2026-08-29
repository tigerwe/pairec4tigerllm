#include "post_rank_hop_burst.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <openssl/sha.h>

namespace pairec::post_rank {
namespace {

int64_t SteadyMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t SystemNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch()).count();
}

void UpdateMax(std::atomic<int>* maximum, int value) {
  int observed = maximum->load(std::memory_order_relaxed);
  while (observed < value && !maximum->compare_exchange_weak(
      observed, value, std::memory_order_relaxed)) {}
}

std::string Escape(const std::string& value) {
  std::ostringstream out;
  for (char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      default: out << ch;
    }
  }
  return out.str();
}

}  // namespace

class BurstCoordinator::Impl {
 public:
  struct Round {
    uint64_t sequence = 0;
    pairec::pipeline::RankRequest request;
    std::mutex mutex;
    std::condition_variable pressure_started_cv;
    int completed = 0;
    int pressure_success = 0;
    int pressure_errors = 0;
    int pressure_overlap_business = 0;
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    std::atomic<int> pressure_started{0};
    int64_t released_us = 0;
    int64_t first_start_us = 0;
    int64_t last_start_us = 0;
    int64_t last_end_us = 0;
    bool business_success = false;
    int64_t business_start_ns = 0;
    int64_t business_end_ns = 0;
    pairec::pipeline::RankResponse business_response;
    std::string business_error;
    double marker_wait_ms = 0;
    std::vector<int64_t> pressure_us;
  };

  struct Worker {
    std::unique_ptr<brpc::Channel> channel;
    std::unique_ptr<pairec::pipeline::DeepFMRankService_Stub> stub;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Round>> rounds;
    std::thread thread;
  };

  ~Impl() { Shutdown(); }

  bool Init(const BurstConfig& config, std::string* error) {
    if (config.business_endpoint.empty() || config.pressure_endpoint.empty() ||
        config.concurrency != 1000 ||
        config.payload_bytes != 102400 || config.business_timeout_ms != 1000 ||
        config.pressure_timeout_ms != 5000 || config.startup_batch_size <= 0 ||
        config.startup_batch_size > config.concurrency ||
        config.pressure_start_quorum < 1 ||
        config.pressure_start_quorum >= config.concurrency ||
        config.pressure_start_timeout_ms < 1) {
      *error = "post-rank hop requires c1000, payload=102400, and a pressure marker quorum";
      return false;
    }
    config_ = config;
    // Keep one preconnected business channel outside the pressure thread pool.
    // The remaining 999 channels are dedicated to pressure Health calls.
    workers_.reserve(config.concurrency);
    for (int lane = 0; lane < config.concurrency; ++lane) {
      workers_.push_back(std::make_unique<Worker>());
    }
    if (!ConnectWorker(*workers_[0], 0)) {
      *error = "post-rank business channel preconnect failed";
      Shutdown();
      return false;
    }
    connected_sessions_ = 1;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config.startup_timeout_ms);
    for (int begin = 1; begin < config.concurrency; begin += config.startup_batch_size) {
      const int end = std::min(config.concurrency, begin + config.startup_batch_size);
      for (int lane = begin; lane < end; ++lane) {
        workers_[lane]->thread = std::thread([this, lane] { WorkerMain(lane); });
      }
      std::unique_lock<std::mutex> lock(state_mutex_);
      if (!startup_cv_.wait_until(lock, deadline, [this, end] {
            return startup_finished_ >= end - 1;
          })) {
        *error = "post-rank hop preconnect startup timed out";
        lock.unlock();
        Shutdown();
        return false;
      }
    }
    if (connected_sessions_ != config.concurrency) {
      *error = "post-rank hop preconnect failed: connected=" +
          std::to_string(connected_sessions_) + "/" + std::to_string(config.concurrency);
      Shutdown();
      return false;
    }
    std::cout << "{\"event\":\"pairec_post_rank_hop2_brpc_burst_ready\","
              << "\"business_endpoint\":\"" << Escape(config.business_endpoint)
              << "\",\"pressure_endpoint\":\"" << Escape(config.pressure_endpoint)
              << "\",\"connected_sessions\":" << connected_sessions_
              << ",\"armed_workers\":" << config.concurrency
              << ",\"pressure_start_quorum\":" << config.pressure_start_quorum
              << ",\"pressure_start_timeout_ms\":" << config.pressure_start_timeout_ms
              << ",\"payload_bytes\":" << config.payload_bytes << "}" << std::endl;
    return true;
  }

  void Shutdown() {
    if (workers_.empty()) return;
    shutdown_.store(true, std::memory_order_release);
    for (auto& worker : workers_) worker->cv.notify_all();
    for (auto& worker : workers_) {
      if (worker->thread.joinable()) worker->thread.join();
    }
    workers_.clear();
  }

  bool Trigger(const pairec::pipeline::RankRequest& request,
               pairec::pipeline::RankResponse* response, BurstResult* result) {
    *result = BurstResult{};
    auto round = std::make_shared<Round>();
    round->request = request;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      round->sequence = ++sequence_;
      ++active_rounds_;
    }
    round->released_us = SteadyMicros();
    std::cout << "{\"event\":\"pairec_post_rank_hop2_brpc_burst_start\","
              << "\"request_id\":\"" << Escape(request.context().request_id())
              << "\",\"concurrency\":" << config_.concurrency
              << ",\"armed_workers\":" << config_.concurrency
              << ",\"pressure_start_quorum\":" << config_.pressure_start_quorum
              << ",\"payload_bytes\":" << config_.payload_bytes << "}" << std::endl;
    {
      std::lock_guard<std::mutex> dispatch_lock(pressure_dispatch_mutex_);
      for (int lane = 1; lane < config_.concurrency; ++lane) {
        Worker& worker = *workers_[lane];
        {
          std::lock_guard<std::mutex> lock(worker.mutex);
          worker.rounds.push_back(round);
        }
        worker.cv.notify_one();
      }
    }
    const int64_t marker_started_us = SteadyMicros();
    bool marker_success = false;
    {
      std::unique_lock<std::mutex> lock(round->mutex);
      marker_success = round->pressure_started_cv.wait_for(
          lock, std::chrono::milliseconds(config_.pressure_start_timeout_ms), [&] {
            return round->pressure_started.load(std::memory_order_acquire) >=
                config_.pressure_start_quorum;
          });
      round->marker_wait_ms = static_cast<double>(SteadyMicros() - marker_started_us) / 1000;
    }
    result->marker_wait_ms = round->marker_wait_ms;
    result->pressure_started_at_business_start =
        round->pressure_started.load(std::memory_order_acquire);

    pairec::pipeline::RankResponse business_response;
    brpc::Controller controller;
    controller.set_timeout_ms(config_.business_timeout_ms);
    const int64_t business_started_ns = SystemNanos();
    {
      std::lock_guard<std::mutex> lock(round->mutex);
      round->business_start_ns = business_started_ns;
    }
    if (marker_success) {
      std::lock_guard<std::mutex> business_lock(business_mutex_);
      workers_[0]->stub->Rank(&controller, &round->request, &business_response, nullptr);
    }
    const int64_t rpc_ended_ns = SystemNanos();
    double service_ms = 0;
    if (business_response.has_trace()) {
      service_ms = static_cast<double>(business_response.trace().total_us()) / 1000;
    }
    const double raw_wall_ms = static_cast<double>(rpc_ended_ns - business_started_ns) / 1e6;
    const double front_brpc_ms = std::max(0.0, raw_wall_ms - service_ms);
    const int64_t business_ended_ns = rpc_ended_ns;
    bool finish = false;
    {
      std::lock_guard<std::mutex> lock(round->mutex);
      round->business_end_ns = business_ended_ns;
      round->business_response = business_response;
      round->business_success = marker_success && !controller.Failed() && business_response.code() == 200;
      if (!marker_success) {
        round->business_error = "pressure start marker timed out: started=" +
            std::to_string(result->pressure_started_at_business_start) +
            " quorum=" + std::to_string(config_.pressure_start_quorum);
      } else if (controller.Failed()) {
        round->business_error = controller.ErrorText();
      } else if (business_response.code() != 200) {
        round->business_error = business_response.message();
      }
      ++round->completed;
      finish = round->completed == config_.concurrency;
      result->success = round->business_success;
      result->business_wall_ms = static_cast<double>(business_ended_ns - business_started_ns) / 1e6;
      result->service_ms = service_ms;
      result->front_brpc_ms = front_brpc_ms;
      result->error = round->business_error;
      *response = round->business_response;
    }
    if (finish) Complete(round);
    std::cout << "{\"event\":\"pairec_post_rank_hop2_brpc_burst_business_complete\","
              << "\"request_id\":\"" << Escape(request.context().request_id())
              << "\",\"concurrency\":" << config_.concurrency
              << ",\"business_success\":" << (result->success ? "true" : "false")
              << ",\"business_client_wall_ms\":" << result->business_wall_ms
              << ",\"service_total_ms\":" << result->service_ms
              << ",\"front_brpc_estimate_ms\":" << result->front_brpc_ms
              << ",\"pressure_start_quorum\":" << config_.pressure_start_quorum
              << ",\"pressure_started_at_business_start\":"
              << result->pressure_started_at_business_start
              << ",\"marker_wait_ms\":" << result->marker_wait_ms
              << "}" << std::endl;
    return result->success;
  }

  bool idle() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return active_rounds_ == 0;
  }

  int connected_sessions() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return connected_sessions_;
  }

  bool ConnectWorker(Worker& worker, int lane) {
    for (int retry = 0; retry <= config_.startup_max_retries; ++retry) {
      brpc::ChannelOptions options;
      options.protocol = "baidu_std";
      options.connection_type = "single";
      options.connection_group = "post_rank_hop2_lane_" + std::to_string(lane);
      options.timeout_ms = config_.pressure_timeout_ms;
      options.max_retry = 0;
      auto channel = std::make_unique<brpc::Channel>();
      const std::string& endpoint = lane == 0 ? config_.business_endpoint : config_.pressure_endpoint;
      if (channel->Init(endpoint.c_str(), "", &options) == 0) {
        auto stub = std::make_unique<pairec::pipeline::DeepFMRankService_Stub>(channel.get());
        pairec::pipeline::HealthRequest request;
        pairec::pipeline::HealthResponse response;
        brpc::Controller controller;
        controller.set_timeout_ms(config_.pressure_timeout_ms);
        stub->Health(&controller, &request, &response, nullptr);
        if (!controller.Failed() && response.code() == 200) {
          worker.channel = std::move(channel);
          worker.stub = std::move(stub);
          return true;
        }
      }
      if (retry < config_.startup_max_retries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(
            config_.startup_retry_backoff_ms * (1 << retry)));
      }
    }
    return false;
  }

  void WorkerMain(int lane) {
    Worker& worker = *workers_[lane];
    const bool connected = ConnectWorker(worker, lane);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      ++startup_finished_;
      if (connected) ++connected_sessions_;
    }
    startup_cv_.notify_all();
    if (!connected) return;

    while (!shutdown_.load(std::memory_order_acquire)) {
      std::shared_ptr<Round> round;
      {
        std::unique_lock<std::mutex> lock(worker.mutex);
        worker.cv.wait(lock, [&] {
          return shutdown_.load(std::memory_order_acquire) || !worker.rounds.empty();
        });
        if (shutdown_.load(std::memory_order_acquire)) return;
        round = worker.rounds.front();
        worker.rounds.pop_front();
      }
      if (shutdown_.load(std::memory_order_acquire)) return;
      const int64_t started_us = SteadyMicros();
      const int64_t started_ns = SystemNanos();
      {
        std::lock_guard<std::mutex> lock(round->mutex);
        round->pressure_started.fetch_add(1, std::memory_order_release);
      }
      round->pressure_started_cv.notify_all();
      const int current = round->active.fetch_add(1) + 1;
      UpdateMax(&round->max_active, current);
      bool ok = false;
      brpc::Controller controller;
      pairec::pipeline::HealthRequest health;
      pairec::pipeline::HealthResponse health_response;
      health.mutable_context()->set_request_id(round->request.context().request_id());
      health.set_payload_padding(std::string(config_.payload_bytes, 'P'));
      controller.set_timeout_ms(config_.pressure_timeout_ms);
      worker.stub->Health(&controller, &health, &health_response, nullptr);
      ok = !controller.Failed() && health_response.code() == 200;
      const int64_t ended_us = SteadyMicros();
      const int64_t ended_ns = SystemNanos();
      round->active.fetch_sub(1);
      bool finish = false;
      {
        std::lock_guard<std::mutex> lock(round->mutex);
        if (round->first_start_us == 0 || started_us < round->first_start_us) round->first_start_us = started_us;
        round->last_start_us = std::max(round->last_start_us, started_us);
        round->last_end_us = std::max(round->last_end_us, ended_us);
        if (ok) ++round->pressure_success;
        else ++round->pressure_errors;
        if (round->business_start_ns > 0 && ended_ns >= round->business_start_ns &&
            (round->business_end_ns == 0 || started_ns <= round->business_end_ns)) {
          ++round->pressure_overlap_business;
        }
        round->pressure_us.push_back(ended_us - started_us);
        ++round->completed;
        finish = round->completed == config_.concurrency;
      }
      if (finish) Complete(round);
    }
  }

  void Complete(const std::shared_ptr<Round>& round) {
    std::vector<int64_t> latencies;
    int success, errors, overlap, max_active;
    int64_t first, last_start, last_end, released;
    bool business_success;
    {
      std::lock_guard<std::mutex> lock(round->mutex);
      latencies = round->pressure_us;
      success = round->pressure_success;
      errors = round->pressure_errors;
      overlap = round->pressure_overlap_business;
      max_active = round->max_active.load();
      first = round->first_start_us;
      last_start = round->last_start_us;
      last_end = round->last_end_us;
      released = round->released_us;
      business_success = round->business_success;
    }
    std::sort(latencies.begin(), latencies.end());
    const double p95 = latencies.empty() ? 0 :
        static_cast<double>(latencies[static_cast<size_t>((latencies.size() - 1) * 0.95)]) / 1000;
    std::cout << "{\"event\":\"pairec_post_rank_hop2_brpc_burst_complete\","
              << "\"request_id\":\"" << Escape(round->request.context().request_id())
              << "\",\"concurrency\":" << config_.concurrency
              << ",\"pressure_requests\":" << config_.concurrency - 1
              << ",\"pressure_success\":" << success
              << ",\"pressure_errors\":" << errors
              << ",\"pressure_overlap_business\":" << overlap
              << ",\"business_success\":" << (business_success ? "true" : "false")
              << ",\"max_active_workers\":" << max_active
              << ",\"start_skew_us\":" << std::max<int64_t>(0, last_start - first)
              << ",\"pressure_latency_p95_ms\":" << p95
              << ",\"burst_total_ms\":" << static_cast<double>(last_end - released) / 1000
              << ",\"burst_valid\":"
              << ((business_success && success == config_.concurrency - 1 && errors == 0) ? "true" : "false")
              << "}" << std::endl;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (active_rounds_ > 0) --active_rounds_;
    }
  }

  BurstConfig config_;
  std::vector<std::unique_ptr<Worker>> workers_;
  mutable std::mutex state_mutex_;
  std::condition_variable startup_cv_;
  std::atomic<bool> shutdown_{false};
  std::mutex business_mutex_;
  std::mutex pressure_dispatch_mutex_;
  int startup_finished_ = 0;
  int connected_sessions_ = 0;
  int active_rounds_ = 0;
  uint64_t sequence_ = 0;
};

BurstCoordinator::BurstCoordinator() : impl_(std::make_unique<Impl>()) {}
BurstCoordinator::~BurstCoordinator() = default;
bool BurstCoordinator::Init(const BurstConfig& config, std::string* error) {
  return impl_->Init(config, error);
}
bool BurstCoordinator::Trigger(const pairec::pipeline::RankRequest& request,
                               pairec::pipeline::RankResponse* response,
                               BurstResult* result) {
  return impl_->Trigger(request, response, result);
}
bool BurstCoordinator::idle() const { return impl_->idle(); }
int BurstCoordinator::connected_sessions() const { return impl_->connected_sessions(); }

std::string OrderedCandidateSha256(
    const google::protobuf::RepeatedPtrField<pairec::pipeline::RankCandidate>& items) {
  SHA256_CTX context;
  SHA256_Init(&context);
  for (const auto& item : items) {
    const uint32_t size = htonl(static_cast<uint32_t>(item.item_id().size()));
    SHA256_Update(&context, &size, sizeof(size));
    SHA256_Update(&context, item.item_id().data(), item.item_id().size());
  }
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_Final(digest, &context);
  std::ostringstream output;
  for (unsigned char byte : digest) {
    output << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
  }
  return output.str();
}

}  // namespace pairec::post_rank
