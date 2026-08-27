#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/time.h>
#include <google/protobuf/message.h>
#include <google/protobuf/unknown_field_set.h>
#include <openssl/sha.h>

#ifdef PAIREC_ENABLE_DATASYSTEM_KV_PROBE
#include <datasystem/kv_client.h>
#endif

#include "kvc_operation_proxy.h"
#include "pipeline_service.pb.h"

namespace {

constexpr int kPayloadPaddingFieldNumber = 101;

struct WrapperConfig {
  int listen_port = 18213;
  std::string backend = "127.0.0.1:18211";
  int backend_timeout_ms = 250;
  int idle_timeout_sec = -1;
  bool rank_kvc_enabled = false;
  std::string rank_kvc_host = "192.168.100.12";
  int rank_kvc_port = 18482;
  std::string rank_kvc_business_key = "rank_feature_context_v1_business";
  uint64_t rank_kvc_object_size = 8388608;
  int rank_kvc_ttl_sec = 600;
  int rank_kvc_business_timeout_ms = 1000;
  int rank_kvc_pressure_timeout_ms = 2000;
  int rank_kvc_registration_timeout_ms = 30000;
  uint64_t rank_kvc_seed = 20260827;
};

bool ParseBool(const std::string& value, bool* output) {
  if (value == "1" || value == "true") {
    *output = true;
    return true;
  }
  if (value == "0" || value == "false") {
    *output = false;
    return true;
  }
  return false;
}

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
    } else if (ConsumeArgValue(argv[i], "rank_kvc_enabled", &value)) {
      if (!ParseBool(value, &config->rank_kvc_enabled)) return false;
    } else if (ConsumeArgValue(argv[i], "rank_kvc_host", &value)) {
      config->rank_kvc_host = value;
    } else if (ConsumeArgValue(argv[i], "rank_kvc_port", &value)) {
      config->rank_kvc_port = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "rank_kvc_business_key", &value)) {
      config->rank_kvc_business_key = value;
    } else if (ConsumeArgValue(argv[i], "rank_kvc_object_size", &value)) {
      config->rank_kvc_object_size = std::strtoull(value.c_str(), nullptr, 10);
    } else if (ConsumeArgValue(argv[i], "rank_kvc_ttl_sec", &value)) {
      config->rank_kvc_ttl_sec = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "rank_kvc_business_timeout_ms", &value)) {
      config->rank_kvc_business_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "rank_kvc_pressure_timeout_ms", &value)) {
      config->rank_kvc_pressure_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "rank_kvc_registration_timeout_ms", &value)) {
      config->rank_kvc_registration_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "rank_kvc_seed", &value)) {
      config->rank_kvc_seed = std::strtoull(value.c_str(), nullptr, 10);
    } else {
      std::cerr << "Unknown argument: " << argv[i] << std::endl;
      return false;
    }
  }
  return config->listen_port > 0 && config->backend_timeout_ms > 0 &&
      !config->backend.empty() && (!config->rank_kvc_enabled ||
      (!config->rank_kvc_host.empty() && config->rank_kvc_port > 0 &&
       config->rank_kvc_port <= 65535 && !config->rank_kvc_business_key.empty() &&
       config->rank_kvc_object_size > 0 && config->rank_kvc_ttl_sec > 0 &&
       config->rank_kvc_business_timeout_ms > 0 &&
       config->rank_kvc_pressure_timeout_ms > 0 &&
       config->rank_kvc_registration_timeout_ms > 0));
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

std::string Sha256(const void* data, size_t size) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(data), size, digest);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned char byte : digest) output << std::setw(2) << static_cast<int>(byte);
  return output.str();
}

std::string DeterministicValue(uint64_t size, uint64_t seed) {
  std::string value(size, '\0');
  std::mt19937_64 random(seed);
  for (uint64_t offset = 0; offset < size; offset += sizeof(uint64_t)) {
    const uint64_t word = random();
    const auto chunk = static_cast<size_t>(
        std::min<uint64_t>(sizeof(word), size - offset));
    std::memcpy(value.data() + offset, &word, chunk);
  }
  return value;
}

struct RankKvcGetResult {
  bool ok = false;
  bool coordinated = false;
  uint64_t bytes = 0;
  double elapsed_ms = 0;
  std::string error;
};

class RankKvcClient {
 public:
  ~RankKvcClient() {
    if (enabled_) pairec::kvc_burst::shutdownInProcessPressureClient();
  }

  bool Init(const WrapperConfig& config) {
    enabled_ = config.rank_kvc_enabled;
    if (!enabled_) return true;
#ifndef PAIREC_ENABLE_DATASYSTEM_KV_PROBE
    std::cerr << "Rank KVC requested but DataSystem support is not compiled" << std::endl;
    return false;
#else
    config_ = config;
    datasystem::ConnectOptions options;
    options.host = config.rank_kvc_host;
    options.port = config.rank_kvc_port;
    options.enableCrossNodeConnection = true;
    options.enableExclusiveConnection = false;
    client_ = std::make_unique<datasystem::KVClient>(options);
    auto status = client_->Init();
    if (status.IsError()) {
      std::cerr << "Rank KVC Init failed: " << status.ToString() << std::endl;
      return false;
    }
    value_ = DeterministicValue(config.rank_kvc_object_size, config.rank_kvc_seed);
    value_sha256_ = Sha256(value_.data(), value_.size());
    datasystem::SetParam param;
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    param.ttlSecond = static_cast<uint32_t>(config.rank_kvc_ttl_sec);
    status = client_->Set(config.rank_kvc_business_key, datasystem::StringView(value_), param);
    if (status.IsError()) {
      std::cerr << "Rank KVC business prefill failed: " << status.ToString() << std::endl;
      return false;
    }
    auto verified = Get(config.rank_kvc_business_key,
                        config.rank_kvc_business_timeout_ms, true);
    if (!verified.ok) {
      std::cerr << "Rank KVC business preflight failed: " << verified.error << std::endl;
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config.rank_kvc_registration_timeout_ms);
    const auto pressure_get = [this](uint32_t, const std::string& key) {
      return Get(key, config_.rank_kvc_pressure_timeout_ms, false).ok;
    };
    while (!pairec::kvc_burst::registerInProcessPressureClient(pressure_get)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::cerr << "Rank KVC pressure client registration timed out" << std::endl;
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::cout << "{\"event\":\"rank_kvc_preflight_complete\",\"host\":\""
              << config.rank_kvc_host << "\",\"port\":" << config.rank_kvc_port
              << ",\"business_key\":\"" << config.rank_kvc_business_key
              << "\",\"object_size_bytes\":" << config.rank_kvc_object_size
              << ",\"ttl_sec\":" << config.rank_kvc_ttl_sec
              << ",\"sha256\":\"" << value_sha256_ << "\"}" << std::endl;
    return true;
#endif
  }

  bool enabled() const { return enabled_; }

  RankKvcGetResult BusinessGet(const std::string& request_id) {
    auto token = pairec::kvc_burst::beginBusinessGet(
        request_id, pairec::kvc_burst::BusinessApi::kGet, 1U);
    auto result = Get(config_.rank_kvc_business_key,
                      config_.rank_kvc_business_timeout_ms, true);
    result.coordinated = token.triggered();
    if (result.coordinated) {
      pairec::kvc_burst::finishBusinessGet(token, result.ok);
    }
    std::cout << "{\"event\":\"rank_kvc_business_get_complete\",\"request_id\":\""
              << request_id << "\",\"success\":" << (result.ok ? "true" : "false")
              << ",\"coordinated\":" << (result.coordinated ? "true" : "false")
              << ",\"trigger_status\":" << static_cast<uint32_t>(token.status)
              << ",\"bytes\":" << result.bytes << ",\"expected_bytes\":"
              << config_.rank_kvc_object_size << ",\"elapsed_ms\":"
              << result.elapsed_ms << ",\"error\":\""
              << (result.error.empty() ? "none" : result.error) << "\"}" << std::endl;
    return result;
  }

 private:
  RankKvcGetResult Get(const std::string& key, int timeout_ms, bool verify_sha) {
    RankKvcGetResult result;
#ifdef PAIREC_ENABLE_DATASYSTEM_KV_PROBE
    butil::Timer timer;
    timer.start();
    datasystem::Optional<datasystem::Buffer> buffer;
    auto status = client_->Get(key, buffer, timeout_ms);
    timer.stop();
    result.elapsed_ms = timer.m_elapsed();
    if (status.IsError()) {
      result.error = status.ToString();
      return result;
    }
    if (!buffer) {
      result.error = "empty DataSystem buffer";
      return result;
    }
    result.bytes = static_cast<uint64_t>(buffer->GetSize());
    if (result.bytes != config_.rank_kvc_object_size) {
      result.error = "short DataSystem buffer: got=" + std::to_string(result.bytes);
      return result;
    }
    if (verify_sha && Sha256(buffer->ImmutableData(), buffer->GetSize()) != value_sha256_) {
      result.error = "DataSystem buffer SHA256 mismatch";
      return result;
    }
    result.ok = true;
#else
    (void) key;
    (void) timeout_ms;
    (void) verify_sha;
    result.error = "DataSystem support is not compiled";
#endif
    return result;
  }

  bool enabled_ = false;
  WrapperConfig config_;
  std::string value_;
  std::string value_sha256_;
#ifdef PAIREC_ENABLE_DATASYSTEM_KV_PROBE
  std::unique_ptr<datasystem::KVClient> client_;
#endif
};

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
  RankBurstWrapper(RankForwarder* forwarder, RankKvcClient* rank_kvc,
                   std::string backend)
      : forwarder_(forwarder), rank_kvc_(rank_kvc), backend_(std::move(backend)) {}

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
    const std::string request_id = request->has_context()
        ? request->context().request_id() : "";
    pairec::pipeline::RankRequest backend_request(*request);
    ClearPayload(&backend_request);
    const int64_t calls_at_backend = health_calls_.load(std::memory_order_relaxed);
    const int64_t bytes_at_backend = health_bytes_.load(std::memory_order_relaxed);
    butil::Timer wrapper_timer;
    wrapper_timer.start();
    RankKvcGetResult rank_kvc_result;
    if (rank_kvc_->enabled()) {
      rank_kvc_result = rank_kvc_->BusinessGet(request_id);
      if (!rank_kvc_result.ok) {
        wrapper_timer.stop();
        active_rank_.fetch_sub(1, std::memory_order_relaxed);
        response->set_code(500);
        response->set_message("rank KVC business Get failed: " + rank_kvc_result.error);
        front->SetFailed(response->message());
        pairec::kvc_burst::observeBusinessRequestComplete(request_id);
        std::cout << "[brpc-rank-burst-wrapper] method=Rank request_id=" << request_id
                  << " code=500 wrapper_total_ms=" << wrapper_timer.m_elapsed()
                  << " backend_rpc_ms=0 backend_service_total_ms=0"
                  << " rank_kvc_get_ms=" << rank_kvc_result.elapsed_ms
                  << " rank_kvc_get_bytes=" << rank_kvc_result.bytes
                  << " rank_kvc_coordinated="
                  << (rank_kvc_result.coordinated ? "true" : "false")
                  << " rank_kvc_success=false backend_skipped=true error="
                  << rank_kvc_result.error << std::endl;
        return;
      }
    }
    double backend_rpc_ms = 0;
    std::string error;
    const bool ok = forwarder_->Rank(
        backend_request, response, &backend_rpc_ms, &error);
    wrapper_timer.stop();
    if (ok && rank_kvc_->enabled() && response->has_trace() &&
        response->trace().has_total_us()) {
      response->mutable_trace()->set_total_us(
          response->trace().total_us() +
          static_cast<int64_t>(rank_kvc_result.elapsed_ms * 1000.0));
    }
    const int64_t ended_epoch_ns = SystemNanos();
    active_rank_.fetch_sub(1, std::memory_order_relaxed);
    pairec::kvc_burst::observeBusinessRequestComplete(request_id);

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
    std::cout << "[brpc-rank-burst-wrapper] method=Rank"
              << " request_id=" << request_id
              << " code=" << response->code()
              << " wrapper_total_ms=" << wrapper_timer.m_elapsed()
              << " backend_rpc_ms=" << backend_rpc_ms
              << " backend_service_total_ms=" << service_total_us / 1000.0
              << " rank_kvc_get_ms=" << rank_kvc_result.elapsed_ms
              << " rank_kvc_get_bytes=" << rank_kvc_result.bytes
              << " rank_kvc_coordinated="
              << (rank_kvc_result.coordinated ? "true" : "false")
              << " rank_kvc_success=" << (rank_kvc_->enabled() ? "true" : "disabled")
              << " backend_skipped=false"
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
    const auto started = std::chrono::steady_clock::now();
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
    auto* trace = response->mutable_trace();
    if (request->has_context()) {
      trace->mutable_context()->CopyFrom(request->context());
    }
    trace->set_component("brpc_rank_burst_wrapper");
    trace->set_protocol("brpc");
    trace->set_status("ok");
    trace->set_total_us(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count());
    trace->set_attribution_complete(true);
    trace->set_asynchronous(false);
  }

 private:
  RankForwarder* forwarder_;
  RankKvcClient* rank_kvc_;
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
  RankKvcClient rank_kvc;
  if (!rank_kvc.Init(config)) {
    std::cerr << "Failed to initialize Rank KVC integration" << std::endl;
    return 1;
  }
  RankBurstWrapper service(&forwarder, &rank_kvc, config.backend);
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
