#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/time.h>

#include "recommend.pb.h"
#include "reverse_burst.h"

namespace {

struct WrapperConfig {
  int listen_port = 18103;
  std::string backend = "127.0.0.1:18100";
  int backend_timeout_ms = 5000;
  int backend_max_retry = 0;
  int idle_timeout_sec = -1;
  pairec::reverse_burst::Config reverse_burst;
};

bool ConsumeArgValue(const char* arg, const std::string& name, std::string* out) {
  const std::string prefix = "--" + name + "=";
  const std::string value(arg);
  if (value.rfind(prefix, 0) != 0) {
    return false;
  }
  *out = value.substr(prefix.size());
  return true;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --listen_port=18103\n"
      << "  --backend=127.0.0.1:18100\n"
      << "  --backend_timeout_ms=5000\n"
      << "  --backend_max_retry=0\n"
      << "  --reverse_burst_endpoint=\n"
      << "  --reverse_burst_stage=generation_return\n"
      << "  --reverse_burst_concurrency=1000\n"
      << "  --reverse_burst_payload_bytes=102400\n"
      << "  --reverse_burst_marker_timeout_ms=1000\n"
      << "  --reverse_burst_pressure_timeout_ms=5000\n"
      << "  --reverse_burst_startup_timeout_ms=120000\n"
      << "  --reverse_burst_startup_batch_size=64\n"
      << "  --reverse_burst_startup_max_retries=3\n"
      << "  --reverse_burst_startup_retry_backoff_ms=100\n"
      << "  --idle_timeout_sec=-1\n";
}

bool ParseArgs(int argc, char** argv, WrapperConfig* config) {
  config->reverse_burst.stage = "generation_return";
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (ConsumeArgValue(argv[i], "listen_port", &value)) {
      config->listen_port = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "backend", &value)) {
      config->backend = value;
    } else if (ConsumeArgValue(argv[i], "backend_timeout_ms", &value)) {
      config->backend_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "backend_max_retry", &value)) {
      config->backend_max_retry = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_endpoint", &value)) {
      config->reverse_burst.endpoint = value;
    } else if (ConsumeArgValue(argv[i], "reverse_burst_stage", &value)) {
      config->reverse_burst.stage = value;
    } else if (ConsumeArgValue(argv[i], "reverse_burst_concurrency", &value)) {
      config->reverse_burst.concurrency = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_payload_bytes", &value)) {
      config->reverse_burst.payload_bytes = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_marker_timeout_ms", &value)) {
      config->reverse_burst.marker_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_pressure_timeout_ms", &value)) {
      config->reverse_burst.pressure_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_startup_timeout_ms", &value)) {
      config->reverse_burst.startup_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_startup_batch_size", &value)) {
      config->reverse_burst.startup_batch_size = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_startup_max_retries", &value)) {
      config->reverse_burst.startup_max_retries = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "reverse_burst_startup_retry_backoff_ms", &value)) {
      config->reverse_burst.startup_retry_backoff_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "idle_timeout_sec", &value)) {
      config->idle_timeout_sec = std::atoi(value.c_str());
    } else if (std::string(argv[i]) == "--help") {
      PrintUsage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (config->listen_port <= 0 || config->backend.empty() ||
      config->backend_timeout_ms <= 0 || config->backend_max_retry < 0) {
    std::cerr << "Invalid wrapper configuration\n";
    PrintUsage(argv[0]);
    return false;
  }
  if (!config->reverse_burst.endpoint.empty() &&
      (config->reverse_burst.stage != "generation_return" ||
       config->reverse_burst.concurrency != 1000 ||
       config->reverse_burst.payload_bytes != 102400 ||
       config->reverse_burst.startup_batch_size <= 0 ||
       config->reverse_burst.startup_max_retries < 0 ||
       config->reverse_burst.startup_retry_backoff_ms < 0)) {
    std::cerr << "Invalid generation reverse burst configuration\n";
    return false;
  }
  return true;
}

void UpdateMax(std::atomic<int64_t>* maximum, int64_t value) {
  int64_t observed = maximum->load(std::memory_order_relaxed);
  while (observed < value &&
         !maximum->compare_exchange_weak(
             observed, value, std::memory_order_relaxed)) {
  }
}

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

bool IsCoordinatedPressure(
    const pairec::inference::HealthRequest& request) {
  static const std::string marker = "PAIREC_BRPC_STOP_V1";
  return request.payload_padding().compare(0, marker.size(), marker) == 0;
}

class ActiveCounterGuard {
 public:
  ActiveCounterGuard(std::atomic<int64_t>* active, std::atomic<int64_t>* maximum)
      : active_(active) {
    const int64_t current = active_->fetch_add(1, std::memory_order_relaxed) + 1;
    UpdateMax(maximum, current);
  }

  ~ActiveCounterGuard() {
    active_->fetch_sub(1, std::memory_order_relaxed);
  }

 private:
  std::atomic<int64_t>* active_;
};

class BackendForwarder {
 public:
  bool Init(const WrapperConfig& config) {
    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.connection_type = "pooled";
    options.timeout_ms = config.backend_timeout_ms;
    options.max_retry = config.backend_max_retry;
    if (channel_.Init(config.backend.c_str(), "", &options) != 0) {
      return false;
    }
    stub_ = std::make_unique<pairec::inference::RecommendService_Stub>(&channel_);
    return true;
  }

  bool Recommend(
      const pairec::inference::RecommendRequest& request,
      pairec::inference::RecommendResponse* response,
      double* backend_rpc_ms,
      std::string* error) {
    brpc::Controller cntl;
    butil::Timer timer;
    timer.start();
    stub_->Recommend(&cntl, &request, response, nullptr);
    timer.stop();
    *backend_rpc_ms = timer.m_elapsed();
    if (cntl.Failed()) {
      *error = cntl.ErrorText();
      return false;
    }
    return true;
  }

 private:
  brpc::Channel channel_;
  std::unique_ptr<pairec::inference::RecommendService_Stub> stub_;
};

class BurstWrapperService final : public pairec::inference::RecommendService {
 public:
  BurstWrapperService(BackendForwarder* forwarder,
                      pairec::reverse_burst::Coordinator* reverse_burst,
                      std::string backend)
      : forwarder_(forwarder), reverse_burst_(reverse_burst),
        backend_(std::move(backend)) {}

  void Recommend(
      google::protobuf::RpcController* controller,
      const pairec::inference::RecommendRequest* request,
      pairec::inference::RecommendResponse* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* front_cntl = static_cast<brpc::Controller*>(controller);
    ActiveCounterGuard total_guard(&active_total_, &max_active_total_);
    max_health_while_recommend_.store(0, std::memory_order_relaxed);
    active_recommend_.fetch_add(1, std::memory_order_relaxed);
    const int64_t recommend_start_epoch_ns = SystemNanos();

    std::cout << "[brpc-burst-wrapper] phase=recommend_start"
              << " request_id=" << request->request_id()
              << " user_id=" << request->user_id()
              << " recommend_start_epoch_ns=" << recommend_start_epoch_ns
              << " active_health=" << active_health_.load(std::memory_order_relaxed)
              << " backend=" << backend_ << std::endl;

    const int64_t health_at_start = active_health_.load(std::memory_order_relaxed);
    const int64_t health_calls_at_start =
        health_calls_.load(std::memory_order_relaxed);
    const int64_t health_payload_bytes_at_start =
        health_payload_bytes_.load(std::memory_order_relaxed);
    const int64_t last_pressure_health_us =
        last_pressure_health_us_.load(std::memory_order_relaxed);
    const bool pressure_drain_required =
        last_pressure_health_us > 0 &&
        SteadyMicros() - last_pressure_health_us <= 100000;
    UpdateMax(&max_health_while_recommend_, health_at_start);
    butil::Timer wrapper_timer;
    wrapper_timer.start();

    bool pressure_drain_success = true;
    double pressure_drain_ms = 0.0;
    double pressure_drain_quiet_ms = 0.0;
    if (pressure_drain_required) {
      const int64_t drain_started_us = SteadyMicros();
      while (true) {
        const int64_t now_us = SteadyMicros();
        const int64_t latest_us =
            last_pressure_health_us_.load(std::memory_order_relaxed);
        pressure_drain_quiet_ms = (now_us - latest_us) / 1000.0;
        pressure_drain_ms = (now_us - drain_started_us) / 1000.0;
        if (pressure_drain_quiet_ms >= 20.0) {
          break;
        }
        if (pressure_drain_ms >= 2000.0) {
          pressure_drain_success = false;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    double backend_rpc_ms = 0.0;
    std::string error;
    const size_t front_payload_bytes = request->payload_padding().size();
    pairec::inference::RecommendRequest backend_request(*request);
    backend_request.clear_payload_padding();
    const int64_t health_calls_at_backend_start =
        health_calls_.load(std::memory_order_relaxed);
    const int64_t health_payload_bytes_at_backend_start =
        health_payload_bytes_.load(std::memory_order_relaxed);
    bool ok = false;
    if (pressure_drain_success) {
      ok = forwarder_->Recommend(
          backend_request, response, &backend_rpc_ms, &error);
    } else {
      error = "BRPC pressure did not drain before backend forwarding";
    }
    pairec::reverse_burst::MarkerResult reverse_result;
    if (ok && response->code() == 200) {
      pairec::reverse_burst::Marker marker;
      marker.request_id = request->request_id();
      marker.backend_code = response->code();
      marker.item_count = response->recommendations_size();
      marker.response_sha256 =
          pairec::reverse_burst::Sha256(response->SerializeAsString());
      if (!reverse_burst_->Trigger(marker, &reverse_result)) {
        ok = false;
        error = "generation reverse burst marker failed: " + reverse_result.error;
      }
    } else if (ok) {
      ok = false;
      error = "generation backend returned code=" + std::to_string(response->code());
    }
    wrapper_timer.stop();
    active_recommend_.fetch_sub(1, std::memory_order_relaxed);

    const int64_t health_calls_during_recommend =
        health_calls_.load(std::memory_order_relaxed) - health_calls_at_start;
    const int64_t health_payload_bytes_during_recommend =
        health_payload_bytes_.load(std::memory_order_relaxed) -
        health_payload_bytes_at_start;
    const int64_t health_calls_during_backend =
        health_calls_.load(std::memory_order_relaxed) -
        health_calls_at_backend_start;
    const int64_t health_payload_bytes_during_backend =
        health_payload_bytes_.load(std::memory_order_relaxed) -
        health_payload_bytes_at_backend_start;
    const int64_t max_health =
        max_health_while_recommend_.load(std::memory_order_relaxed);
    const double wrapper_total_ms = wrapper_timer.m_elapsed();
    const double backend_inference_ms = response->inference_time_ms();
    if (!ok) {
      response->set_code(500);
      response->set_user_id(request->user_id());
      response->set_error("backend brpc Recommend failed: " + error);
      front_cntl->SetFailed(response->error());
    }
    auto* trace = response->mutable_trace();
    trace->set_wrapper_total_ms(wrapper_total_ms);
    trace->set_wrapper_backend_rpc_ms(backend_rpc_ms);
    trace->set_wrapper_overhead_ms(wrapper_total_ms - backend_rpc_ms);
    trace->set_wrapper_active_health_at_start(health_at_start);
    trace->set_wrapper_max_active_health(max_health);
    trace->set_wrapper_max_active_total(max_health + 1);
    trace->set_wrapper_backend_brpc_ms(backend_rpc_ms - backend_inference_ms);

    std::cout << "[brpc-burst-wrapper] method=Recommend"
              << " request_id=" << request->request_id()
              << " user_id=" << request->user_id()
              << " code=" << response->code()
              << " wrapper_total_ms=" << wrapper_total_ms
              << " backend_rpc_ms=" << backend_rpc_ms
              << " backend_inference_ms=" << backend_inference_ms
              << " wrapper_overhead_ms=" << wrapper_total_ms - backend_rpc_ms
              << " backend_brpc_ms=" << backend_rpc_ms - backend_inference_ms
              << " active_health_at_start=" << health_at_start
              << " max_active_health=" << max_health
              << " max_active_total=" << max_health + 1
              << " recommend_start_epoch_ns=" << recommend_start_epoch_ns
              << " pressure_drain_required=" << pressure_drain_required
              << " pressure_drain_success=" << pressure_drain_success
              << " pressure_drain_ms=" << pressure_drain_ms
              << " pressure_drain_quiet_ms=" << pressure_drain_quiet_ms
              << " health_calls_at_start=" << health_calls_at_start
              << " health_calls_during_recommend="
              << health_calls_during_recommend
              << " health_payload_bytes_during_recommend="
              << health_payload_bytes_during_recommend
              << " health_calls_during_backend="
              << health_calls_during_backend
              << " health_payload_bytes_during_backend="
              << health_payload_bytes_during_backend
              << " front_payload_bytes=" << front_payload_bytes
              << " backend_payload_bytes=" << backend_request.payload_padding().size()
              << " reverse_burst_enabled=" << reverse_result.enabled
              << " reverse_marker_success=" << reverse_result.success
              << " reverse_marker_wall_ms=" << reverse_result.wall_ms
              << " reverse_marker_sink_ms=" << reverse_result.sink_ms
              << " reverse_marker_front_ms=" << reverse_result.front_ms
              << " backend=" << backend_
              << " error=" << (error.empty() ? "none" : error) << std::endl;
  }

  void Health(
      google::protobuf::RpcController*,
      const pairec::inference::HealthRequest* request,
      pairec::inference::HealthResponse* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::string action;
    if (pairec::reverse_burst::ParseControl(request->payload_padding(), &action)) {
      std::string error;
      bool success = true;
      if (action == "arm") success = reverse_burst_->Arm(&error);
      if (action == "disarm") success = reverse_burst_->Disarm(&error);
      response->set_code(success ? 200 : 409);
      response->set_status(reverse_burst_->armed() ? "armed" : "disarmed");
      response->set_backend("brpc_burst_wrapper");
      response->set_raw_json(
          "{\"action\":\"" + action + "\",\"success\":" +
          (success ? "true" : "false") + ",\"armed\":" +
          (reverse_burst_->armed() ? "true" : "false") +
          ",\"idle\":" + (reverse_burst_->idle() ? "true" : "false") +
          ",\"connected_sessions\":" +
          std::to_string(reverse_burst_->connected_sessions()) +
          ",\"error\":\"" + error + "\"}");
      return;
    }
    ActiveCounterGuard total_guard(&active_total_, &max_active_total_);
    ActiveCounterGuard health_guard(&active_health_, &max_active_health_);
    health_calls_.fetch_add(1, std::memory_order_relaxed);
    health_payload_bytes_.fetch_add(
        static_cast<int64_t>(request->payload_padding().size()),
        std::memory_order_relaxed);
    if (IsCoordinatedPressure(*request)) {
      last_pressure_health_us_.store(SteadyMicros(), std::memory_order_relaxed);
    }
    const int64_t health = active_health_.load(std::memory_order_relaxed);
    if (active_recommend_.load(std::memory_order_relaxed) > 0) {
      UpdateMax(&max_health_while_recommend_, health);
    }

    response->set_code(200);
    response->set_status("healthy");
    response->set_backend("brpc_burst_wrapper");
    response->set_raw_json(
        "{\"payload_bytes\":" + std::to_string(request->payload_padding().size()) +
        ",\"backend_forwarded\":false}");
  }

 private:
  BackendForwarder* forwarder_;
  pairec::reverse_burst::Coordinator* reverse_burst_;
  std::string backend_;
  std::atomic<int64_t> active_total_{0};
  std::atomic<int64_t> max_active_total_{0};
  std::atomic<int64_t> active_health_{0};
  std::atomic<int64_t> max_active_health_{0};
  std::atomic<int64_t> active_recommend_{0};
  std::atomic<int64_t> max_health_while_recommend_{0};
  std::atomic<int64_t> health_calls_{0};
  std::atomic<int64_t> health_payload_bytes_{0};
  std::atomic<int64_t> last_pressure_health_us_{0};
};

}  // namespace

int main(int argc, char** argv) {
  WrapperConfig config;
  if (!ParseArgs(argc, argv, &config)) {
    return 2;
  }

  BackendForwarder forwarder;
  if (!forwarder.Init(config)) {
    std::cerr << "Failed to initialize backend brpc channel to "
              << config.backend << std::endl;
    return 1;
  }

  pairec::reverse_burst::Coordinator reverse_burst;
  std::string reverse_error;
  if (!reverse_burst.Init(config.reverse_burst, &reverse_error)) {
    std::cerr << "Failed to initialize generation reverse burst: "
              << reverse_error << std::endl;
    return 1;
  }

  BurstWrapperService service(&forwarder, &reverse_burst, config.backend);
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    std::cerr << "Failed to add RecommendService" << std::endl;
    return 1;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = config.idle_timeout_sec;
  if (server.Start(config.listen_port, &options) != 0) {
    std::cerr << "Failed to start brpc burst wrapper on port "
              << config.listen_port << std::endl;
    return 1;
  }

  std::cout << "brpc burst wrapper listening on 0.0.0.0:"
            << config.listen_port << ", forwarding Recommend to "
            << config.backend << ", terminating Health locally" << std::endl;
  server.RunUntilAskedToQuit();
  return 0;
}
