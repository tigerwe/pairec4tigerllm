#include "reverse_burst.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/time.h>
#include <openssl/sha.h>

#include "recommend.pb.h"

namespace pairec::reverse_burst {
namespace {

int64_t SteadyMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t SystemNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(character) << std::dec;
        } else {
          output << character;
        }
    }
  }
  return output.str();
}

void UpdateMax(std::atomic<int>* maximum, int value) {
  int observed = maximum->load(std::memory_order_relaxed);
  while (observed < value && !maximum->compare_exchange_weak(
      observed, value, std::memory_order_relaxed)) {}
}

std::string BuildPayload(const Config& config, const Marker& marker,
                         const char* kind, int lane) {
  std::ostringstream header;
  header << "PAIREC_RETURN_BURST_V1\n"
         << "kind=" << kind << "\n"
         << "stage=" << config.stage << "\n"
         << "request_id=" << marker.request_id << "\n"
         << "backend_code=" << marker.backend_code << "\n"
         << "item_count=" << marker.item_count << "\n"
         << "response_sha256=" << marker.response_sha256 << "\n"
         << "lane=" << lane << "\n"
         << "header_end=1\n";
  std::string payload = header.str();
  if (payload.size() > static_cast<size_t>(config.payload_bytes)) return {};
  payload.resize(static_cast<size_t>(config.payload_bytes), 'R');
  return payload;
}

double ExtractSinkMs(const std::string& raw_json) {
  const std::string key = "\"service_us\":";
  const auto begin = raw_json.find(key);
  if (begin == std::string::npos) return 0;
  const auto value_begin = begin + key.size();
  const auto value_end = raw_json.find_first_not_of("0123456789", value_begin);
  try {
    return std::stod(raw_json.substr(value_begin, value_end - value_begin)) / 1000.0;
  } catch (...) {
    return 0;
  }
}

bool SinkEchoMatches(const std::string& raw_json, const Marker& marker) {
  return raw_json.find("\"request_id\":\"" + marker.request_id + "\"") !=
             std::string::npos &&
      raw_json.find("\"response_sha256\":\"" + marker.response_sha256 + "\"") !=
             std::string::npos;
}

}  // namespace

class Coordinator::Impl {
 public:
  struct Round {
    uint64_t sequence = 0;
    Marker marker;
    std::mutex mutex;
    std::condition_variable cv;
    int ready_workers = 0;
    bool released = false;
    int completed = 0;
    int success = 0;
    int errors = 0;
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    int64_t first_start_us = 0;
    int64_t last_start_us = 0;
    int64_t last_end_us = 0;
    int64_t first_start_epoch_ns = 0;
    int64_t last_end_epoch_ns = 0;
    int64_t marker_end_epoch_ns = 0;
    std::vector<double> pressure_latencies_ms;
    bool marker_done = false;
    bool marker_success = false;
    double marker_wall_ms = 0;
    double marker_sink_ms = 0;
    std::string marker_error;
  };

  struct Worker {
    std::unique_ptr<brpc::Channel> channel;
    std::unique_ptr<pairec::inference::RecommendService_Stub> stub;
    std::thread thread;
  };

  ~Impl() { Shutdown(); }

  bool Init(const Config& config, std::string* error) {
    if (config.endpoint.empty()) return true;
    if (config.stage.empty() || config.concurrency != 1000 ||
        config.payload_bytes != 102400 || config.marker_timeout_ms <= 0 ||
        config.pressure_timeout_ms <= 0 || config.startup_timeout_ms <= 0) {
      *error = "reverse burst requires stage, concurrency=1000, payload_bytes=102400, and positive timeouts";
      return false;
    }
    config_ = config;
    enabled_.store(true, std::memory_order_release);
    armed_.store(config.initially_armed, std::memory_order_release);
    workers_.reserve(static_cast<size_t>(config.concurrency));
    for (int lane = 0; lane < config.concurrency; ++lane) {
      auto worker = std::make_unique<Worker>();
      worker->channel = std::make_unique<brpc::Channel>();
      workers_.push_back(std::move(worker));
    }
    try {
      for (int lane = 0; lane < config.concurrency; ++lane) {
        workers_[lane]->thread = std::thread([this, lane] { WorkerMain(lane); });
      }
    } catch (const std::exception& exception) {
      *error = "reverse burst worker creation failed: " +
          std::string(exception.what());
      Shutdown();
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config.startup_timeout_ms);
    std::unique_lock<std::mutex> lock(state_mutex_);
    startup_cv_.wait_until(lock, deadline, [this] {
      return startup_finished_ == config_.concurrency;
    });
    if (connected_sessions_ != config_.concurrency) {
      *error = "reverse burst preconnect failed: connected=" +
          std::to_string(connected_sessions_) + "/" +
          std::to_string(config_.concurrency);
      lock.unlock();
      Shutdown();
      return false;
    }
    std::cout << "{\"event\":\"pairec_reverse_brpc_burst_ready\",\"stage\":\""
              << JsonEscape(config_.stage) << "\",\"endpoint\":\""
              << JsonEscape(config_.endpoint) << "\",\"connected_sessions\":"
              << connected_sessions_ << ",\"armed_workers\":" << config_.concurrency
              << ",\"connection_groups\":" << config_.concurrency
              << ",\"payload_bytes\":" << config_.payload_bytes
              << ",\"initially_armed\":"
              << (config_.initially_armed ? "true" : "false") << "}" << std::endl;
    return true;
  }

  void Shutdown() {
    if (workers_.empty()) return;
    shutdown_.store(true, std::memory_order_release);
    state_cv_.notify_all();
    std::shared_ptr<Round> round;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      round = current_round_;
    }
    if (round) round->cv.notify_all();
    for (auto& worker : workers_) {
      if (worker->thread.joinable()) worker->thread.join();
    }
    workers_.clear();
  }

  bool Arm(std::string* error) {
    if (!enabled()) {
      *error = "reverse burst endpoint is not configured";
      return false;
    }
    if (!idle()) {
      *error = "reverse burst round is still active";
      return false;
    }
    armed_.store(true, std::memory_order_release);
    LogControl("arm");
    return true;
  }

  bool Disarm(std::string*) {
    armed_.store(false, std::memory_order_release);
    LogControl("disarm");
    return true;
  }

  bool Trigger(const Marker& marker, MarkerResult* result) {
    *result = MarkerResult{};
    result->enabled = enabled() && armed();
    if (!result->enabled) {
      result->success = true;
      return true;
    }
    if (marker.request_id.empty() || marker.response_sha256.size() != 64) {
      result->error = "invalid reverse marker identity";
      return false;
    }
    auto round = std::make_shared<Round>();
    round->marker = marker;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (current_round_) {
        result->error = "previous reverse burst round is still active";
        return false;
      }
      round->sequence = ++sequence_;
      current_round_ = round;
    }
    state_cv_.notify_all();

    const auto ready_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config_.marker_timeout_ms);
    {
      std::unique_lock<std::mutex> lock(round->mutex);
      if (!round->cv.wait_until(lock, ready_deadline, [this, &round] {
            return round->ready_workers == config_.concurrency;
          })) {
        result->error = "timed out arming reverse burst workers";
        round->released = true;
        round->cv.notify_all();
        return false;
      }
      std::cout << "{\"event\":\"pairec_reverse_brpc_burst_start\",\"stage\":\""
                << JsonEscape(config_.stage) << "\",\"request_id\":\""
                << JsonEscape(marker.request_id) << "\",\"concurrency\":"
                << config_.concurrency << ",\"armed_workers\":"
                << round->ready_workers << ",\"payload_bytes\":"
                << config_.payload_bytes << "}" << std::endl;
      round->released = true;
      round->cv.notify_all();
    }

    const auto marker_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config_.marker_timeout_ms);
    std::unique_lock<std::mutex> lock(round->mutex);
    if (!round->cv.wait_until(lock, marker_deadline, [&round] {
          return round->marker_done;
        })) {
      result->error = "reverse burst marker timed out";
      return false;
    }
    result->success = round->marker_success;
    result->wall_ms = round->marker_wall_ms;
    result->sink_ms = round->marker_sink_ms;
    result->front_ms = std::max(0.0, result->wall_ms - result->sink_ms);
    result->error = round->marker_error;
    return result->success;
  }

  bool enabled() const { return enabled_.load(std::memory_order_acquire); }
  bool armed() const { return armed_.load(std::memory_order_acquire); }
  bool idle() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_round_ == nullptr;
  }

  void WorkerMain(int lane) {
    Worker& worker = *workers_[lane];
    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.connection_type = "single";
    // Single-server Channels share a Socket when connection_group is equal.
    // A group per lane makes the 1000 advertised sessions real and keeps the
    // marker off the pressure lanes' unwritten-byte buffers.
    options.connection_group = config_.stage + "_lane_" + std::to_string(lane);
    options.timeout_ms = config_.pressure_timeout_ms;
    options.max_retry = 0;
    bool connected = worker.channel->Init(config_.endpoint.c_str(), "", &options) == 0;
    if (connected) {
      worker.stub = std::make_unique<pairec::inference::RecommendService_Stub>(
          worker.channel.get());
      pairec::inference::HealthRequest request;
      pairec::inference::HealthResponse response;
      brpc::Controller controller;
      worker.stub->Health(&controller, &request, &response, nullptr);
      connected = !controller.Failed() && response.code() == 200;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      ++startup_finished_;
      if (connected) ++connected_sessions_;
    }
    startup_cv_.notify_all();
    if (!connected) return;

    uint64_t seen_sequence = 0;
    while (!shutdown_.load(std::memory_order_acquire)) {
      std::shared_ptr<Round> round;
      {
        std::unique_lock<std::mutex> lock(state_mutex_);
        state_cv_.wait(lock, [this, seen_sequence] {
          return shutdown_.load(std::memory_order_acquire) ||
              (current_round_ && current_round_->sequence > seen_sequence);
        });
        if (shutdown_.load(std::memory_order_acquire)) break;
        round = current_round_;
        seen_sequence = round->sequence;
      }
      {
        std::unique_lock<std::mutex> lock(round->mutex);
        ++round->ready_workers;
        round->cv.notify_all();
        round->cv.wait(lock, [this, &round] {
          return shutdown_.load(std::memory_order_acquire) || round->released;
        });
      }
      if (shutdown_.load(std::memory_order_acquire)) break;
      Execute(lane, worker, round);
    }
  }

  void Execute(int lane, Worker& worker, const std::shared_ptr<Round>& round) {
    const int active = round->active.fetch_add(1, std::memory_order_relaxed) + 1;
    UpdateMax(&round->max_active, active);
    const int64_t start_us = SteadyMicros();
    const int64_t start_epoch_ns = SystemNanos();
    bool success = false;
    std::string error;
    double wall_ms = 0;
    double sink_ms = 0;
    const std::string payload = BuildPayload(
        config_, round->marker, lane == 0 ? "marker" : "health", lane);
    if (payload.empty()) {
      error = "reverse burst metadata exceeds fixed payload";
    } else {
      butil::Timer timer;
      timer.start();
      brpc::Controller controller;
      controller.set_timeout_ms(
          lane == 0 ? config_.marker_timeout_ms : config_.pressure_timeout_ms);
      if (lane == 0) {
        pairec::inference::RecommendRequest request;
        request.set_user_id(config_.stage);
        request.set_request_id(round->marker.request_id);
        request.set_topk(round->marker.item_count);
        request.set_beam_width(round->marker.backend_code);
        request.set_payload_padding(payload);
        pairec::inference::RecommendResponse response;
        worker.stub->Recommend(&controller, &request, &response, nullptr);
        timer.stop();
        wall_ms = timer.m_elapsed();
        sink_ms = ExtractSinkMs(response.raw_json());
        success = !controller.Failed() && response.code() == 200 &&
            response.user_id() == config_.stage &&
            SinkEchoMatches(response.raw_json(), round->marker);
        if (!success) {
          error = controller.Failed() ? controller.ErrorText() : response.error();
          if (error.empty()) error = "reverse marker identity echo mismatch";
        }
      } else {
        pairec::inference::HealthRequest request;
        request.set_payload_padding(payload);
        pairec::inference::HealthResponse response;
        worker.stub->Health(&controller, &request, &response, nullptr);
        timer.stop();
        wall_ms = timer.m_elapsed();
        success = !controller.Failed() && response.code() == 200;
        if (!success) error = controller.Failed() ? controller.ErrorText() : response.status();
      }
    }
    const int64_t end_us = SteadyMicros();
    const int64_t end_epoch_ns = SystemNanos();
    round->active.fetch_sub(1, std::memory_order_relaxed);
    bool complete = false;
    {
      std::lock_guard<std::mutex> lock(round->mutex);
      if (round->first_start_us == 0 || start_us < round->first_start_us) {
        round->first_start_us = start_us;
      }
      round->last_start_us = std::max(round->last_start_us, start_us);
      round->last_end_us = std::max(round->last_end_us, end_us);
      if (round->first_start_epoch_ns == 0 ||
          start_epoch_ns < round->first_start_epoch_ns) {
        round->first_start_epoch_ns = start_epoch_ns;
      }
      round->last_end_epoch_ns = std::max(round->last_end_epoch_ns, end_epoch_ns);
      ++round->completed;
      if (success) ++round->success; else ++round->errors;
      if (lane == 0) {
        round->marker_end_epoch_ns = end_epoch_ns;
        round->marker_done = true;
        round->marker_success = success;
        round->marker_wall_ms = wall_ms;
        round->marker_sink_ms = sink_ms;
        round->marker_error = error;
        std::cout << "{\"event\":\"pairec_reverse_brpc_burst_marker_complete\",\"stage\":\""
                  << JsonEscape(config_.stage) << "\",\"request_id\":\""
                  << JsonEscape(round->marker.request_id) << "\",\"success\":"
                  << (success ? "true" : "false") << ",\"marker_wall_ms\":"
                  << wall_ms << ",\"marker_sink_ms\":" << sink_ms
                  << ",\"marker_front_ms\":" << std::max(0.0, wall_ms - sink_ms)
                  << ",\"marker_end_epoch_ns\":" << end_epoch_ns
                  << ",\"error\":\"" << JsonEscape(error) << "\"}" << std::endl;
      } else {
        round->pressure_latencies_ms.push_back(wall_ms);
      }
      complete = round->completed == config_.concurrency;
      round->cv.notify_all();
    }
    if (complete) FinishRound(round);
  }

  void FinishRound(const std::shared_ptr<Round>& round) {
    int completed;
    int success;
    int errors;
    int64_t first_start_us;
    int64_t last_start_us;
    int64_t last_end_us;
    int64_t first_start_epoch_ns;
    int64_t last_end_epoch_ns;
    int64_t marker_end_epoch_ns;
    std::vector<double> pressure_latencies_ms;
    {
      std::lock_guard<std::mutex> lock(round->mutex);
      completed = round->completed;
      success = round->success;
      errors = round->errors;
      first_start_us = round->first_start_us;
      last_start_us = round->last_start_us;
      last_end_us = round->last_end_us;
      first_start_epoch_ns = round->first_start_epoch_ns;
      last_end_epoch_ns = round->last_end_epoch_ns;
      marker_end_epoch_ns = round->marker_end_epoch_ns;
      pressure_latencies_ms = round->pressure_latencies_ms;
    }
    std::sort(pressure_latencies_ms.begin(), pressure_latencies_ms.end());
    const auto percentile = [&pressure_latencies_ms](double q) {
      if (pressure_latencies_ms.empty()) return 0.0;
      const size_t index = static_cast<size_t>(
          q * static_cast<double>(pressure_latencies_ms.size() - 1));
      return pressure_latencies_ms[index];
    };
    std::cout << "{\"event\":\"pairec_reverse_brpc_burst_complete\",\"stage\":\""
              << JsonEscape(config_.stage) << "\",\"request_id\":\""
              << JsonEscape(round->marker.request_id) << "\",\"concurrency\":"
              << config_.concurrency << ",\"marker_success\":"
              << (round->marker_success ? "true" : "false")
              << ",\"pressure_requests\":" << config_.concurrency - 1
              << ",\"pressure_success\":" << success - (round->marker_success ? 1 : 0)
              << ",\"pressure_errors\":" << errors - (round->marker_success ? 0 : 1)
              << ",\"completed\":" << completed << ",\"accepted_bytes\":"
              << static_cast<int64_t>(success) * config_.payload_bytes
              << ",\"max_active\":" << round->max_active.load(std::memory_order_relaxed)
              << ",\"pressure_latency_p95_ms\":" << percentile(0.95)
              << ",\"pressure_latency_max_ms\":"
              << (pressure_latencies_ms.empty() ? 0.0 : pressure_latencies_ms.back())
              << ",\"start_skew_us\":" << last_start_us - first_start_us
              << ",\"burst_total_ms\":" << (last_end_us - first_start_us) / 1000.0
              << ",\"tail_after_marker_ms\":"
              << std::max<int64_t>(0, last_end_epoch_ns - marker_end_epoch_ns) / 1e6
              << ",\"first_start_epoch_ns\":" << first_start_epoch_ns
              << ",\"marker_end_epoch_ns\":" << marker_end_epoch_ns
              << ",\"last_end_epoch_ns\":" << last_end_epoch_ns
              << "}" << std::endl;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (current_round_ == round) current_round_.reset();
    }
    idle_cv_.notify_all();
  }

  void LogControl(const char* action) const {
    std::cout << "{\"event\":\"pairec_reverse_brpc_burst_control\",\"stage\":\""
              << JsonEscape(config_.stage) << "\",\"action\":\"" << action
              << "\",\"armed\":" << (armed() ? "true" : "false")
              << ",\"idle\":" << (idle() ? "true" : "false") << "}" << std::endl;
  }

  Config config_;
  std::vector<std::unique_ptr<Worker>> workers_;
  mutable std::mutex state_mutex_;
  std::condition_variable state_cv_;
  std::condition_variable startup_cv_;
  std::condition_variable idle_cv_;
  std::shared_ptr<Round> current_round_;
  uint64_t sequence_ = 0;
  int startup_finished_ = 0;
  int connected_sessions_ = 0;
  std::atomic<bool> enabled_{false};
  std::atomic<bool> armed_{false};
  std::atomic<bool> shutdown_{false};
};

Coordinator::Coordinator() : impl_(std::make_unique<Impl>()) {}
Coordinator::~Coordinator() = default;
bool Coordinator::Init(const Config& config, std::string* error) {
  return impl_->Init(config, error);
}
bool Coordinator::Arm(std::string* error) { return impl_->Arm(error); }
bool Coordinator::Disarm(std::string* error) { return impl_->Disarm(error); }
bool Coordinator::Trigger(const Marker& marker, MarkerResult* result) {
  return impl_->Trigger(marker, result);
}
bool Coordinator::enabled() const { return impl_->enabled(); }
bool Coordinator::armed() const { return impl_->armed(); }
bool Coordinator::idle() const { return impl_->idle(); }
int Coordinator::connected_sessions() const { return impl_->connected_sessions_; }
int Coordinator::concurrency() const { return impl_->config_.concurrency; }
int Coordinator::payload_bytes() const { return impl_->config_.payload_bytes; }
const std::string& Coordinator::stage() const { return impl_->config_.stage; }

bool ParseControl(const std::string& payload, std::string* action) {
  if (payload.compare(0, sizeof(kControlPrefix) - 1, kControlPrefix) != 0) return false;
  *action = payload.substr(sizeof(kControlPrefix) - 1);
  return *action == "arm" || *action == "disarm" || *action == "status";
}

std::string Sha256(const std::string& value) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(), digest);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned char byte : digest) output << std::setw(2) << static_cast<int>(byte);
  return output.str();
}

}  // namespace pairec::reverse_burst
