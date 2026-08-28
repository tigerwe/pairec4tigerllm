#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <set>
#include <string>
#include <utility>

#include <brpc/server.h>

#include "recommend.pb.h"

namespace {

struct Config {
  int listen_port = 18301;
  std::string stage = "generation_return";
  int expected_payload_bytes = 102400;
  int expected_requests = 1000;
  int idle_timeout_sec = -1;
};

bool Consume(const char* arg, const std::string& name, std::string* output) {
  const std::string prefix = "--" + name + "=";
  const std::string input(arg == nullptr ? "" : arg);
  if (input.compare(0, prefix.size(), prefix) != 0) return false;
  *output = input.substr(prefix.size());
  return true;
}

bool ParseArgs(int argc, char** argv, Config* config) {
  for (int index = 1; index < argc; ++index) {
    std::string value;
    if (Consume(argv[index], "listen_port", &value)) {
      config->listen_port = std::atoi(value.c_str());
    } else if (Consume(argv[index], "stage", &value)) {
      config->stage = value;
    } else if (Consume(argv[index], "expected_payload_bytes", &value)) {
      config->expected_payload_bytes = std::atoi(value.c_str());
    } else if (Consume(argv[index], "expected_requests", &value)) {
      config->expected_requests = std::atoi(value.c_str());
    } else if (Consume(argv[index], "idle_timeout_sec", &value)) {
      config->idle_timeout_sec = std::atoi(value.c_str());
    } else {
      std::cerr << "Unknown argument: " << argv[index] << std::endl;
      return false;
    }
  }
  return config->listen_port > 0 && !config->stage.empty() &&
      config->expected_payload_bytes == 102400 && config->expected_requests == 1000;
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

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    if (character == '\\') output << "\\\\";
    else if (character == '"') output << "\\\"";
    else if (character == '\n') output << "\\n";
    else if (character >= 0x20) output << character;
  }
  return output.str();
}

bool IsSha256(const std::string& value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

struct PayloadMetadata {
  std::string kind;
  std::string stage;
  std::string request_id;
  std::string response_sha256;
  int backend_code = 0;
  int item_count = 0;
  int lane = -1;
};

bool ParseInteger(const std::string& value, int* output) {
  try {
    size_t used = 0;
    const int parsed = std::stoi(value, &used);
    if (used != value.size()) return false;
    *output = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParsePayload(const std::string& payload, PayloadMetadata* metadata,
                  std::string* error) {
  static const std::string prefix = "PAIREC_RETURN_BURST_V1\n";
  if (payload.compare(0, prefix.size(), prefix) != 0) {
    *error = "missing payload contract prefix";
    return false;
  }
  const auto end = payload.find("header_end=1\n", prefix.size());
  if (end == std::string::npos) {
    *error = "missing payload header terminator";
    return false;
  }
  std::map<std::string, std::string> fields;
  std::istringstream input(payload.substr(prefix.size(), end - prefix.size()));
  std::string line;
  while (std::getline(input, line)) {
    const auto equals = line.find('=');
    if (equals == std::string::npos) continue;
    fields[line.substr(0, equals)] = line.substr(equals + 1);
  }
  metadata->kind = fields["kind"];
  metadata->stage = fields["stage"];
  metadata->request_id = fields["request_id"];
  metadata->response_sha256 = fields["response_sha256"];
  if ((metadata->kind != "marker" && metadata->kind != "health") ||
      metadata->stage.empty() || metadata->request_id.empty() ||
      !IsSha256(metadata->response_sha256) ||
      !ParseInteger(fields["backend_code"], &metadata->backend_code) ||
      !ParseInteger(fields["item_count"], &metadata->item_count) ||
      !ParseInteger(fields["lane"], &metadata->lane)) {
    *error = "invalid payload metadata";
    return false;
  }
  if ((metadata->kind == "marker" && metadata->lane != 0) ||
      (metadata->kind == "health" && metadata->lane <= 0)) {
    *error = "payload kind/lane mismatch";
    return false;
  }
  return true;
}

class ReturnPressureSink final : public pairec::inference::RecommendService {
 public:
  explicit ReturnPressureSink(Config config) : config_(std::move(config)) {}

  class ActiveGuard {
   public:
    explicit ActiveGuard(std::atomic<int>* active)
        : active_(active), current_(active_->fetch_add(1) + 1) {}
    ~ActiveGuard() { active_->fetch_sub(1); }
    int current() const { return current_; }
   private:
    std::atomic<int>* active_;
    int current_;
  };

  void Recommend(google::protobuf::RpcController* controller,
                 const pairec::inference::RecommendRequest* request,
                 pairec::inference::RecommendResponse* response,
                 google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    ActiveGuard active_guard(&active_);
    const int64_t started_us = SteadyMicros();
    PayloadMetadata metadata;
    std::string error;
    bool valid = Validate(request->payload_padding(), "marker", &metadata, &error);
    valid = valid && request->request_id() == metadata.request_id &&
        request->user_id() == metadata.stage && request->topk() == metadata.item_count &&
        request->beam_width() == metadata.backend_code;
    if (!valid && error.empty()) error = "marker protobuf metadata mismatch";
    const int64_t service_us = SteadyMicros() - started_us;
    if (!valid) {
      response->set_code(400);
      response->set_error(error);
      static_cast<brpc::Controller*>(controller)->SetFailed(error);
      Observe(metadata, false, request->payload_padding().size(), service_us,
              active_guard.current());
      return;
    }
    response->set_code(200);
    response->set_user_id(config_.stage);
    response->set_raw_json(
        "{\"service_us\":" + std::to_string(service_us) +
        ",\"accepted_bytes\":" +
        std::to_string(request->payload_padding().size()) +
        ",\"stage\":\"" + JsonEscape(config_.stage) +
        "\",\"request_id\":\"" + JsonEscape(metadata.request_id) +
        "\",\"response_sha256\":\"" +
        JsonEscape(metadata.response_sha256) + "\"}");
    std::cout << "{\"event\":\"pairec_return_sink_marker_complete\",\"stage\":\""
              << JsonEscape(config_.stage) << "\",\"request_id\":\""
              << JsonEscape(metadata.request_id) << "\",\"payload_valid\":true"
              << ",\"backend_code\":" << metadata.backend_code
              << ",\"item_count\":" << metadata.item_count
              << ",\"response_sha256\":\"" << metadata.response_sha256
              << "\",\"accepted_bytes\":" << request->payload_padding().size()
              << ",\"service_us\":" << service_us << "}" << std::endl;
    Observe(metadata, true, request->payload_padding().size(), service_us,
            active_guard.current());
  }

  void Health(google::protobuf::RpcController* controller,
              const pairec::inference::HealthRequest* request,
              pairec::inference::HealthResponse* response,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (request->payload_padding().empty()) {
      response->set_code(200);
      response->set_status("healthy");
      response->set_backend("brpc_return_pressure_sink");
      return;
    }
    ActiveGuard active_guard(&active_);
    const int64_t started_us = SteadyMicros();
    PayloadMetadata metadata;
    std::string error;
    const bool valid = Validate(request->payload_padding(), "health", &metadata, &error);
    const int64_t service_us = SteadyMicros() - started_us;
    if (!valid) {
      response->set_code(400);
      response->set_status("invalid");
      static_cast<brpc::Controller*>(controller)->SetFailed(error);
      Observe(metadata, false, request->payload_padding().size(), service_us,
              active_guard.current());
      return;
    }
    response->set_code(200);
    response->set_status("healthy");
    response->set_backend("brpc_return_pressure_sink");
    Observe(metadata, true, request->payload_padding().size(), service_us,
            active_guard.current());
  }

 private:
  struct Round {
    std::string response_sha256;
    int backend_code = 0;
    int item_count = 0;
    int received = 0;
    int marker_count = 0;
    int health_count = 0;
    int errors = 0;
    int64_t accepted_bytes = 0;
    int active = 0;
    int max_active = 0;
    std::set<int> lanes;
    int64_t first_us = 0;
    int64_t last_us = 0;
    int64_t first_epoch_ns = 0;
    int64_t marker_end_epoch_ns = 0;
    int64_t last_epoch_ns = 0;
    bool completed_logged = false;
  };

  bool Validate(const std::string& payload, const char* expected_kind,
                PayloadMetadata* metadata, std::string* error) const {
    if (payload.size() != static_cast<size_t>(config_.expected_payload_bytes)) {
      *error = "payload size mismatch";
      return false;
    }
    if (!ParsePayload(payload, metadata, error)) return false;
    if (metadata->kind != expected_kind || metadata->stage != config_.stage ||
        metadata->lane < 0 || metadata->lane >= config_.expected_requests) {
      *error = "payload routing metadata mismatch";
      return false;
    }
    return true;
  }

  void Observe(const PayloadMetadata& metadata, bool valid, size_t bytes,
               int64_t service_us, int observed_active) {
    if (metadata.request_id.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& round = rounds_[metadata.request_id];
    const int64_t now_us = SteadyMicros();
    const int64_t now_epoch_ns = SystemNanos();
    if (round.first_us == 0) {
      round.first_us = now_us;
      round.first_epoch_ns = now_epoch_ns;
      round.response_sha256 = metadata.response_sha256;
      round.backend_code = metadata.backend_code;
      round.item_count = metadata.item_count;
    }
    round.max_active = std::max(round.max_active, observed_active);
    ++round.received;
    if (metadata.kind == "marker") ++round.marker_count;
    if (metadata.kind == "marker") round.marker_end_epoch_ns = now_epoch_ns;
    if (metadata.kind == "health") ++round.health_count;
    const bool unique_lane = round.lanes.insert(metadata.lane).second;
    const bool identity_valid = unique_lane &&
        metadata.response_sha256 == round.response_sha256 &&
        metadata.backend_code == round.backend_code &&
        metadata.item_count == round.item_count;
    if (!valid || !identity_valid) ++round.errors;
    if (valid && identity_valid) round.accepted_bytes += static_cast<int64_t>(bytes);
    round.last_us = now_us;
    round.last_epoch_ns = std::max(round.last_epoch_ns, now_epoch_ns);
    if (round.received == config_.expected_requests && !round.completed_logged) {
      round.completed_logged = true;
      std::cout << "{\"event\":\"pairec_return_sink_burst_complete\",\"stage\":\""
                << JsonEscape(config_.stage) << "\",\"request_id\":\""
                << JsonEscape(metadata.request_id) << "\",\"received\":"
                << round.received << ",\"marker_count\":" << round.marker_count
                << ",\"health_count\":" << round.health_count
                << ",\"unique_lanes\":" << round.lanes.size()
                << ",\"errors\":" << round.errors
                << ",\"accepted_bytes\":" << round.accepted_bytes
                << ",\"payload_bytes\":" << config_.expected_payload_bytes
                << ",\"payload_valid\":" << (round.errors == 0 ? "true" : "false")
                << ",\"max_active\":" << round.max_active
                << ",\"total_ms\":" << (round.last_us - round.first_us) / 1000.0
                << ",\"first_epoch_ns\":" << round.first_epoch_ns
                << ",\"marker_end_epoch_ns\":" << round.marker_end_epoch_ns
                << ",\"last_end_epoch_ns\":" << round.last_epoch_ns
                << ",\"response_sha256\":\"" << round.response_sha256 << "\"}"
                << std::endl;
    }
    if (rounds_.size() > 64) rounds_.erase(rounds_.begin());
  }

  Config config_;
  std::atomic<int> active_{0};
  std::mutex mutex_;
  std::map<std::string, Round> rounds_;
};

}  // namespace

int main(int argc, char** argv) {
  Config config;
  if (!ParseArgs(argc, argv, &config)) return 2;
  ReturnPressureSink service(config);
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) return 1;
  brpc::ServerOptions options;
  options.idle_timeout_sec = config.idle_timeout_sec;
  if (server.Start(config.listen_port, &options) != 0) return 1;
  std::cout << "{\"event\":\"pairec_return_sink_ready\",\"stage\":\""
            << JsonEscape(config.stage) << "\",\"listen_port\":"
            << config.listen_port << ",\"expected_requests\":"
            << config.expected_requests << ",\"payload_bytes\":"
            << config.expected_payload_bytes << "}" << std::endl;
  server.RunUntilAskedToQuit();
  return 0;
}
