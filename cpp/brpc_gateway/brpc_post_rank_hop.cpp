#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <brpc/controller.h>
#include <brpc/server.h>
#include <google/protobuf/stubs/callback.h>

#if __has_include("pipeline_service.pb.h")
#include "pipeline_service.pb.h"
#else
#include "pairec_ub_probe/pipeline_service.pb.h"
#endif
#include "post_rank_hop_burst.h"
#include "ubsocket_trace_key_workaround.h"

#ifndef PAIREC_SOURCE_COMMIT
#define PAIREC_SOURCE_COMMIT "unknown"
#endif

namespace {

struct Config {
  int listen_port = 18311;
  int pressure_listen_port = 18313;
  std::string role = "hop1";
  pairec::post_rank::BurstConfig burst;
  int idle_timeout_sec = -1;
  std::string transport = "tcp";
};

bool Value(const char* arg, const std::string& name, std::string* output) {
  const std::string prefix = "--" + name + "=";
  const std::string value(arg == nullptr ? "" : arg);
  if (value.compare(0, prefix.size(), prefix) != 0) return false;
  *output = value.substr(prefix.size());
  return true;
}

bool Parse(int argc, char** argv, Config* config) {
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (Value(argv[i], "listen_port", &value)) config->listen_port = std::atoi(value.c_str());
    else if (Value(argv[i], "role", &value)) config->role = value;
    else if (Value(argv[i], "backend", &value)) config->burst.business_endpoint = value;
    else if (Value(argv[i], "business_backend", &value)) config->burst.business_endpoint = value;
    else if (Value(argv[i], "pressure_backend", &value)) config->burst.pressure_endpoint = value;
    else if (Value(argv[i], "concurrency", &value)) config->burst.concurrency = std::atoi(value.c_str());
    else if (Value(argv[i], "payload_bytes", &value)) config->burst.payload_bytes = std::atoi(value.c_str());
    else if (Value(argv[i], "business_timeout_ms", &value)) config->burst.business_timeout_ms = std::atoi(value.c_str());
    else if (Value(argv[i], "pressure_timeout_ms", &value)) config->burst.pressure_timeout_ms = std::atoi(value.c_str());
    else if (Value(argv[i], "pressure_start_quorum", &value)) config->burst.pressure_start_quorum = std::atoi(value.c_str());
    else if (Value(argv[i], "pressure_start_timeout_ms", &value)) config->burst.pressure_start_timeout_ms = std::atoi(value.c_str());
    else if (Value(argv[i], "startup_timeout_ms", &value)) config->burst.startup_timeout_ms = std::atoi(value.c_str());
    else if (Value(argv[i], "startup_batch_size", &value)) config->burst.startup_batch_size = std::atoi(value.c_str());
    else if (Value(argv[i], "startup_max_retries", &value)) config->burst.startup_max_retries = std::atoi(value.c_str());
    else if (Value(argv[i], "startup_retry_backoff_ms", &value)) config->burst.startup_retry_backoff_ms = std::atoi(value.c_str());
    else if (Value(argv[i], "idle_timeout_sec", &value)) config->idle_timeout_sec = std::atoi(value.c_str());
    else if (Value(argv[i], "pressure_listen_port", &value)) config->pressure_listen_port = std::atoi(value.c_str());
    else if (Value(argv[i], "transport", &value)) config->transport = value;
    else {
      std::cerr << "Unknown argument: " << argv[i] << std::endl;
      return false;
    }
  }
  return config->listen_port > 0 && config->pressure_listen_port > 0 &&
      (config->role == "hop1" || config->role == "hop2") &&
      (config->transport == "tcp" || config->transport == "ub") &&
      (config->role != "hop1" ||
       (!config->burst.business_endpoint.empty() && !config->burst.pressure_endpoint.empty()));
}

int64_t SystemNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string Escape(const std::string& value) {
  std::ostringstream out;
  for (char ch : value) {
    if (ch == '\\' || ch == '"') out << '\\';
    out << ch;
  }
  return out.str();
}

class PostRankHopService final : public pairec::pipeline::DeepFMRankService {
 public:
  PostRankHopService(std::string role, pairec::post_rank::BurstCoordinator* burst)
      : role_(std::move(role)), burst_(burst) {}

  void Rank(google::protobuf::RpcController* controller,
            const pairec::pipeline::RankRequest* request,
            pairec::pipeline::RankResponse* response,
            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    const auto started = std::chrono::steady_clock::now();
    if (request == nullptr || !request->has_context() || request->context().request_id().empty() ||
        request->items_size() != 50 || request->payload_padding().size() != 102400) {
      response->set_code(400);
      response->set_message("post-rank hop contract violation");
      return;
    }
    const std::string input_sha = pairec::post_rank::OrderedCandidateSha256(request->items());
    pairec::post_rank::BurstResult burst_result;
    if (role_ == "hop1") {
      if (burst_ == nullptr || !burst_->Trigger(*request, response, &burst_result)) {
        response->Clear();
        response->set_code(500);
        response->set_message("post-rank hop2 burst failed: " + burst_result.error);
        return;
      }
    } else {
      response->set_code(200);
      response->set_message("success");
      response->set_model_version("post-rank-hop-v1");
      response->set_model_role("passthrough");
      for (const auto& candidate : request->items()) {
        auto* item = response->add_items();
        item->set_item_id(candidate.item_id());
        item->set_score(0);
      }
    }
    if (response->items_size() != request->items_size()) {
      response->Clear();
      response->set_code(500);
      response->set_message("post-rank item count changed");
      return;
    }
    for (int index = 0; index < request->items_size(); ++index) {
      if (response->items(index).item_id() != request->items(index).item_id()) {
        response->Clear();
        response->set_code(500);
        response->set_message("post-rank item order changed");
        return;
      }
    }
    const auto ended = std::chrono::steady_clock::now();
    const int64_t total_us = std::chrono::duration_cast<std::chrono::microseconds>(ended - started).count();
    auto* trace = response->mutable_trace();
    trace->mutable_context()->CopyFrom(request->context());
    trace->set_component(role_ == "hop1" ? "post_rank_hop1" : "post_rank_hop2");
    trace->set_protocol("brpc");
    trace->set_status("ok");
    trace->set_total_us(total_us);
    trace->set_backend_total_us(static_cast<int64_t>(burst_result.business_wall_ms * 1000));
    trace->set_backend_rpc_us(static_cast<int64_t>(
        burst_result.front_brpc_ms * 1000));
    std::cout << "{\"event\":\"post_rank_" << role_ << "_business_complete\","
              << "\"request_id\":\"" << Escape(request->context().request_id())
              << "\",\"candidate_count\":" << request->items_size()
              << ",\"candidate_sha256\":\"" << input_sha
              << "\",\"payload_bytes\":" << request->payload_padding().size()
              << ",\"service_total_ms\":" << std::fixed << std::setprecision(3)
              << static_cast<double>(total_us) / 1000
              << ",\"strict_order_valid\":true}" << std::endl;
  }

  void Health(google::protobuf::RpcController*,
              const pairec::pipeline::HealthRequest* request,
              pairec::pipeline::HealthResponse* response,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    response->set_code(200);
    response->set_status("healthy");
    response->set_backend("post_rank_" + role_);
    auto* trace = response->mutable_trace();
    if (request != nullptr && request->has_context()) {
      trace->mutable_context()->CopyFrom(request->context());
    }
    trace->set_component("post_rank_" + role_);
    trace->set_protocol("brpc");
    trace->set_status("ok");
    trace->set_total_us(1);
  }

 private:
  std::string role_;
  pairec::post_rank::BurstCoordinator* burst_;
};

}  // namespace

int main(int argc, char** argv) {
  Config config;
  if (!Parse(argc, argv, &config)) return 2;
  config.burst.use_ub = config.transport == "ub";
  if (config.transport == "ub" &&
      !pairec::brpc_ub_probe::InitializeUBSocketTraceKeys(
          "PAIREC_POST_RANK_UB_TRACE_KEYS_READY")) {
    return 1;
  }
  std::cout << "{\"event\":\"pairec_post_rank_binary_identity\","
            << "\"source_commit\":\"" << PAIREC_SOURCE_COMMIT << "\","
            << "\"role\":\"" << config.role << "\","
            << "\"transport\":\"" << config.transport << "\"}" << std::endl;
  pairec::post_rank::BurstCoordinator burst;
  if (config.role == "hop1") {
    std::string error;
    if (!burst.Init(config.burst, &error)) {
      std::cerr << "Failed to initialize post-rank hop2 burst: " << error << std::endl;
      return 1;
    }
  }
  PostRankHopService service(config.role, config.role == "hop1" ? &burst : nullptr);
  std::unique_ptr<PostRankHopService> pressure_service;
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    std::cerr << "Failed to register post-rank " << config.role
              << " business service" << std::endl;
    return 1;
  }
  brpc::ServerOptions options;
  options.idle_timeout_sec = config.idle_timeout_sec;
#if defined(BRPC_WITH_URMA)
  // Hop-1's listener remains TCP for the Go caller. Only Hop-2 accepts the
  // c1000 UB fan-out from Hop-1.
  options.use_ub = config.role == "hop2" && config.transport == "ub";
#else
  if (config.transport == "ub") {
    std::cerr << "UB transport requested but this binary lacks BRPC_WITH_URMA"
              << std::endl;
    return 1;
  }
#endif
  if (server.Start(config.listen_port, &options) != 0) {
    std::cerr << "Failed to start post-rank " << config.role
              << " business listener on port " << config.listen_port << std::endl;
    return 1;
  }
  brpc::Server pressure_server;
  if (config.role == "hop2") {
    pressure_service = std::make_unique<PostRankHopService>(config.role, nullptr);
    if (pressure_server.AddService(
            pressure_service.get(), brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
      std::cerr << "Failed to register post-rank hop2 pressure service" << std::endl;
      return 1;
    }
    if (pressure_server.Start(config.pressure_listen_port, &options) != 0) {
      std::cerr << "Failed to start post-rank hop2 pressure listener on port "
                << config.pressure_listen_port << std::endl;
      return 1;
    }
    std::cout << "post-rank hop2 pressure listening on 0.0.0.0:"
              << config.pressure_listen_port << std::endl;
  }
  std::cout << "post-rank " << config.role << " listening on 0.0.0.0:"
            << config.listen_port << std::endl;
  server.RunUntilAskedToQuit();
  if (config.role == "hop2") pressure_server.Stop(0);
  return 0;
}
