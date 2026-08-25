#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/time.h>
#include <google/protobuf/message.h>
#include <google/protobuf/unknown_field_set.h>

#include "pipeline_service.pb.h"

namespace {

constexpr int kPayloadPaddingFieldNumber = 101;

struct WrapperConfig {
  int listen_port = 18213;
  std::string backend = "127.0.0.1:18211";
  int backend_timeout_ms = 250;
  int idle_timeout_sec = -1;
};

bool ConsumeArgValue(const char* arg, const std::string& name, std::string* out) {
  const std::string prefix = "--" + name + "=";
  const std::string value(arg);
  if (value.rfind(prefix, 0) != 0) return false;
  *out = value.substr(prefix.size());
  return true;
}

bool ParseArgs(int argc, char** argv, WrapperConfig* config) {
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (ConsumeArgValue(argv[i], "listen_port", &value)) {
      config->listen_port = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "backend", &value)) {
      config->backend = value;
    } else if (ConsumeArgValue(argv[i], "backend_timeout_ms", &value)) {
      config->backend_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "idle_timeout_sec", &value)) {
      config->idle_timeout_sec = std::atoi(value.c_str());
    } else {
      std::cerr << "Unknown argument: " << argv[i] << std::endl;
      return false;
    }
  }
  return config->listen_port > 0 && config->backend_timeout_ms > 0 &&
      !config->backend.empty();
}

int64_t SystemNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void UpdateMax(std::atomic<int64_t>* maximum, int64_t value) {
  int64_t observed = maximum->load(std::memory_order_relaxed);
  while (observed < value && !maximum->compare_exchange_weak(
      observed, value, std::memory_order_relaxed)) {}
}

class ActiveGuard {
 public:
  ActiveGuard(std::atomic<int64_t>* active, std::atomic<int64_t>* maximum)
      : active_(active) {
    const int64_t current = active_->fetch_add(1, std::memory_order_relaxed) + 1;
    UpdateMax(maximum, current);
  }
  ~ActiveGuard() { active_->fetch_sub(1, std::memory_order_relaxed); }

 private:
  std::atomic<int64_t>* active_;
};

size_t PayloadBytes(const google::protobuf::Message& message) {
  const auto* descriptor = message.GetDescriptor();
  const auto* reflection = message.GetReflection();
  const auto* field = descriptor->FindFieldByNumber(kPayloadPaddingFieldNumber);
  if (field != nullptr && field->type() == google::protobuf::FieldDescriptor::TYPE_BYTES &&
      reflection->HasField(message, field)) {
    return reflection->GetString(message, field).size();
  }
  size_t bytes = 0;
  const auto& unknown = reflection->GetUnknownFields(message);
  for (int index = 0; index < unknown.field_count(); ++index) {
    const auto& item = unknown.field(index);
    if (item.number() == kPayloadPaddingFieldNumber &&
        item.type() == google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
      bytes += item.length_delimited().size();
    }
  }
  return bytes;
}

void ClearPayload(google::protobuf::Message* message) {
  const auto* descriptor = message->GetDescriptor();
  const auto* reflection = message->GetReflection();
  const auto* field = descriptor->FindFieldByNumber(kPayloadPaddingFieldNumber);
  if (field != nullptr) reflection->ClearField(message, field);
  reflection->MutableUnknownFields(message)->DeleteByNumber(kPayloadPaddingFieldNumber);
}

class RankForwarder {
 public:
  bool Init(const WrapperConfig& config) {
    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.connection_type = "pooled";
    options.timeout_ms = config.backend_timeout_ms;
    options.max_retry = 0;
    if (channel_.Init(config.backend.c_str(), "", &options) != 0) return false;
    stub_ = std::make_unique<pairec::pipeline::DeepFMRankService_Stub>(&channel_);
    return true;
  }

  bool Rank(const pairec::pipeline::RankRequest& request,
            pairec::pipeline::RankResponse* response,
            double* elapsed_ms, std::string* error) {
    brpc::Controller controller;
    butil::Timer timer;
    timer.start();
    stub_->Rank(&controller, &request, response, nullptr);
    timer.stop();
    *elapsed_ms = timer.m_elapsed();
    if (controller.Failed()) {
      *error = controller.ErrorText();
      return false;
    }
    return true;
  }

 private:
  brpc::Channel channel_;
  std::unique_ptr<pairec::pipeline::DeepFMRankService_Stub> stub_;
};

class RankBurstWrapper final : public pairec::pipeline::DeepFMRankService {
 public:
  RankBurstWrapper(RankForwarder* forwarder, std::string backend)
      : forwarder_(forwarder), backend_(std::move(backend)) {}

  void Rank(google::protobuf::RpcController* controller,
            const pairec::pipeline::RankRequest* request,
            pairec::pipeline::RankResponse* response,
            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    auto* front = static_cast<brpc::Controller*>(controller);
    ActiveGuard total_guard(&active_total_, &max_active_total_);
    active_rank_.fetch_add(1, std::memory_order_relaxed);
    max_health_while_rank_.store(0, std::memory_order_relaxed);
    const int64_t started_epoch_ns = SystemNanos();
    const int64_t health_at_start = active_health_.load(std::memory_order_relaxed);
    const int64_t calls_at_start = health_calls_.load(std::memory_order_relaxed);
    const int64_t bytes_at_start = health_bytes_.load(std::memory_order_relaxed);
    UpdateMax(&max_health_while_rank_, health_at_start);

    const size_t front_payload_bytes = PayloadBytes(*request);
    pairec::pipeline::RankRequest backend_request(*request);
    ClearPayload(&backend_request);
    const int64_t calls_at_backend = health_calls_.load(std::memory_order_relaxed);
    const int64_t bytes_at_backend = health_bytes_.load(std::memory_order_relaxed);
    butil::Timer wrapper_timer;
    wrapper_timer.start();
    double backend_rpc_ms = 0;
    std::string error;
    const bool ok = forwarder_->Rank(
        backend_request, response, &backend_rpc_ms, &error);
    wrapper_timer.stop();
    const int64_t ended_epoch_ns = SystemNanos();
    active_rank_.fetch_sub(1, std::memory_order_relaxed);

    if (!ok) {
      response->set_code(500);
      response->set_message("rank backend failed: " + error);
      front->SetFailed(response->message());
    }
    const int64_t calls_during_rank =
        health_calls_.load(std::memory_order_relaxed) - calls_at_start;
    const int64_t bytes_during_rank =
        health_bytes_.load(std::memory_order_relaxed) - bytes_at_start;
    const int64_t calls_during_backend =
        health_calls_.load(std::memory_order_relaxed) - calls_at_backend;
    const int64_t bytes_during_backend =
        health_bytes_.load(std::memory_order_relaxed) - bytes_at_backend;
    const int64_t service_total_us = response->has_trace() && response->trace().has_total_us()
        ? response->trace().total_us() : 0;
    const std::string request_id = request->has_context()
        ? request->context().request_id() : "";
    std::cout << "[brpc-rank-burst-wrapper] method=Rank"
              << " request_id=" << request_id
              << " code=" << response->code()
              << " wrapper_total_ms=" << wrapper_timer.m_elapsed()
              << " backend_rpc_ms=" << backend_rpc_ms
              << " backend_service_total_ms=" << service_total_us / 1000.0
              << " active_health_at_start=" << health_at_start
              << " max_active_health=" << max_health_while_rank_.load(std::memory_order_relaxed)
              << " max_active_total=" << max_active_total_.load(std::memory_order_relaxed)
              << " rank_start_epoch_ns=" << started_epoch_ns
              << " rank_end_epoch_ns=" << ended_epoch_ns
              << " health_calls_at_start=" << calls_at_start
              << " health_calls_during_rank=" << calls_during_rank
              << " health_payload_bytes_during_rank=" << bytes_during_rank
              << " health_calls_during_backend=" << calls_during_backend
              << " health_payload_bytes_during_backend=" << bytes_during_backend
              << " front_payload_bytes=" << front_payload_bytes
              << " backend_payload_bytes=" << PayloadBytes(backend_request)
              << " backend=" << backend_
              << " error=" << (error.empty() ? "none" : error) << std::endl;
  }

  void Health(google::protobuf::RpcController*,
              const pairec::pipeline::HealthRequest* request,
              pairec::pipeline::HealthResponse* response,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    ActiveGuard total_guard(&active_total_, &max_active_total_);
    ActiveGuard health_guard(&active_health_, &max_active_health_);
    const size_t bytes = PayloadBytes(*request);
    health_calls_.fetch_add(1, std::memory_order_relaxed);
    health_bytes_.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
    if (active_rank_.load(std::memory_order_relaxed) > 0) {
      UpdateMax(&max_health_while_rank_, active_health_.load(std::memory_order_relaxed));
    }
    response->set_code(200);
    response->set_status("healthy");
    response->set_backend("brpc_rank_burst_wrapper");
  }

 private:
  RankForwarder* forwarder_;
  std::string backend_;
  std::atomic<int64_t> active_total_{0};
  std::atomic<int64_t> max_active_total_{0};
  std::atomic<int64_t> active_health_{0};
  std::atomic<int64_t> max_active_health_{0};
  std::atomic<int64_t> active_rank_{0};
  std::atomic<int64_t> max_health_while_rank_{0};
  std::atomic<int64_t> health_calls_{0};
  std::atomic<int64_t> health_bytes_{0};
};

}  // namespace

int main(int argc, char** argv) {
  WrapperConfig config;
  if (!ParseArgs(argc, argv, &config)) return 2;
  RankForwarder forwarder;
  if (!forwarder.Init(config)) {
    std::cerr << "Failed to initialize Rank backend " << config.backend << std::endl;
    return 1;
  }
  RankBurstWrapper service(&forwarder, config.backend);
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) return 1;
  brpc::ServerOptions options;
  options.idle_timeout_sec = config.idle_timeout_sec;
  if (server.Start(config.listen_port, &options) != 0) return 1;
  std::cout << "brpc Rank burst wrapper listening on 0.0.0.0:" << config.listen_port
            << " backend=" << config.backend << std::endl;
  server.RunUntilAskedToQuit();
  return 0;
}
