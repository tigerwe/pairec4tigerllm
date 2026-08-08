#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <google/protobuf/util/json_util.h>

#include "pipeline_service.pb.h"

namespace {

using Clock = std::chrono::steady_clock;

int64_t ElapsedUs(const Clock::time_point& started) {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count();
}

struct Config {
#if defined(PAIREC_VECTOR_ADAPTER)
  int listen_port = 18201;
  std::string backend = "127.0.0.1:18200";
  int backend_timeout_ms = 250;
#else
  int listen_port = 18211;
  std::string backend = "127.0.0.1:18210";
  int backend_timeout_ms = 80;
#endif
  int idle_timeout_sec = -1;
};

bool Consume(const char* arg, const std::string& name, std::string* value) {
  const std::string prefix = "--" + name + "=";
  const std::string input(arg);
  if (input.rfind(prefix, 0) != 0) {
    return false;
  }
  *value = input.substr(prefix.size());
  return true;
}

bool ParseArgs(int argc, char** argv, Config* config) {
  for (int index = 1; index < argc; ++index) {
    std::string value;
    if (Consume(argv[index], "listen_port", &value)) {
      config->listen_port = std::atoi(value.c_str());
    } else if (Consume(argv[index], "backend", &value)) {
      config->backend = value;
    } else if (Consume(argv[index], "backend_timeout_ms", &value)) {
      config->backend_timeout_ms = std::atoi(value.c_str());
    } else if (Consume(argv[index], "idle_timeout_sec", &value)) {
      config->idle_timeout_sec = std::atoi(value.c_str());
    } else {
      std::cerr << "Unknown argument: " << argv[index] << std::endl;
      return false;
    }
  }
  return config->listen_port > 0 && config->backend_timeout_ms > 0 && !config->backend.empty();
}

class LocalBackend {
 public:
  bool Init(const Config& config) {
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.connection_type = "pooled";
    options.timeout_ms = config.backend_timeout_ms;
    options.max_retry = 0;
    return channel_.Init(config.backend.c_str(), "", &options) == 0;
  }

  bool Post(const std::string& path, const std::string& body, const std::string& request_id,
            std::string* response, int* status, std::string* error) {
    brpc::Controller controller;
    controller.http_request().uri() = path;
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.http_request().SetHeader("Content-Type", "application/json");
    controller.http_request().SetHeader("X-Request-ID", request_id);
    controller.request_attachment().append(body);
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
    if (controller.Failed()) {
      *status = 0;
      *error = controller.ErrorText();
      return false;
    }
    *status = controller.http_response().status_code();
    controller.response_attachment().copy_to(response);
    return *status >= 200 && *status < 300;
  }

  bool Get(const std::string& path, std::string* response, int* status, std::string* error) {
    brpc::Controller controller;
    controller.http_request().uri() = path;
    controller.http_request().set_method(brpc::HTTP_METHOD_GET);
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
    if (controller.Failed()) {
      *status = 0;
      *error = controller.ErrorText();
      return false;
    }
    *status = controller.http_response().status_code();
    controller.response_attachment().copy_to(response);
    return *status >= 200 && *status < 300;
  }

 private:
  brpc::Channel channel_;
};

template <typename Request>
bool ValidateContext(const Request& request, std::string* error) {
  if (!request.has_context() || request.context().request_id().empty()) {
    *error = "trace context request_id is required";
    return false;
  }
  if (request.context().contract_version() != "pairec.pipeline_trace.v1") {
    *error = "unsupported trace contract version";
    return false;
  }
  return true;
}

template <typename Request, typename Response>
bool Forward(LocalBackend* backend, const std::string& path, const Request& request,
             Response* response, int64_t* backend_rpc_us, std::string* error) {
  google::protobuf::util::JsonPrintOptions print_options;
  print_options.preserve_proto_field_names = true;
  std::string body;
  const auto print_status = google::protobuf::util::MessageToJsonString(request, &body, print_options);
  if (!print_status.ok()) {
    *error = "request JSON encode failed: " + print_status.ToString();
    return false;
  }
  std::string response_body;
  int status = 0;
  const auto backend_started = Clock::now();
  const bool posted = backend->Post(path, body, request.context().request_id(),
                                    &response_body, &status, error);
  *backend_rpc_us = ElapsedUs(backend_started);
  if (!posted) {
    if (error->empty()) {
      *error = "backend HTTP status=" + std::to_string(status);
    }
    return false;
  }
  google::protobuf::util::JsonParseOptions parse_options;
  parse_options.ignore_unknown_fields = true;
  const auto parse_status =
      google::protobuf::util::JsonStringToMessage(response_body, response, parse_options);
  if (!parse_status.ok()) {
    *error = "backend JSON decode failed: " + parse_status.ToString();
    return false;
  }
  return true;
}

template <typename Request, typename Response>
void CompleteTrace(const Request& request, Response* response, const std::string& component,
                   const Clock::time_point& started, int64_t backend_rpc_us,
                   const std::string& status, const std::string& error) {
  auto* trace = response->mutable_trace();
  if (request.has_context()) {
    trace->mutable_context()->CopyFrom(request.context());
  }
  trace->set_component(component);
  trace->set_protocol("brpc");
  trace->set_status(status);
  trace->set_backend_rpc_us(backend_rpc_us);
  trace->set_total_us(ElapsedUs(started));
  trace->set_attribution_complete(true);
  trace->set_asynchronous(false);
  if (!error.empty()) {
    trace->set_attribution_reason(error);
  }
}

void CompleteHealth(LocalBackend* backend, const pairec::pipeline::HealthRequest* request,
                    pairec::pipeline::HealthResponse* response, const std::string& component,
                    google::protobuf::RpcController* controller) {
  const auto started = Clock::now();
  std::string body;
  std::string error;
  int status = 0;
  bool ok = backend->Get("/health", &body, &status, &error);
#if defined(PAIREC_VECTOR_ADAPTER)
  if (ok && body.find("\"milvus\":true") == std::string::npos) {
    ok = false;
    error = "vector backend health does not confirm milvus=true";
  }
#else
  if (ok && body.find("\"status\":\"healthy\"") == std::string::npos) {
    ok = false;
    error = "rank backend health does not report healthy";
  }
#endif
  response->set_code(ok ? 200 : 503);
  response->set_status(ok ? "healthy" : "unhealthy");
  response->set_backend(component);
  auto* trace = response->mutable_trace();
  if (request->has_context()) {
    trace->mutable_context()->CopyFrom(request->context());
  }
  trace->set_component(component);
  trace->set_protocol("brpc");
  trace->set_status(ok ? "ok" : "error");
  trace->set_backend_rpc_us(ElapsedUs(started));
  trace->set_total_us(ElapsedUs(started));
  trace->set_attribution_complete(ok);
  if (!ok) {
    static_cast<brpc::Controller*>(controller)->SetFailed(error);
  }
}

#if defined(PAIREC_VECTOR_ADAPTER)
class VectorService final : public pairec::pipeline::VectorRecallService {
 public:
  explicit VectorService(LocalBackend* backend) : backend_(backend) {}

  void Recall(google::protobuf::RpcController* controller,
              const pairec::pipeline::VectorRecallRequest* request,
              pairec::pipeline::VectorRecallResponse* response,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    const auto started = Clock::now();
    std::string error;
    int64_t backend_rpc_us = 0;
    bool ok = ValidateContext(*request, &error);
    if (ok) {
      ok = Forward(backend_, "/recall", *request, response, &backend_rpc_us, &error);
    }
    if (!ok) {
      response->set_code(500);
      response->set_message(error);
    }
    CompleteTrace(*request, response, "vector_recall_adapter", started, backend_rpc_us,
                  ok ? "ok" : "error", error);
    if (!ok) {
      static_cast<brpc::Controller*>(controller)->SetFailed(error);
    }
  }

  void Health(google::protobuf::RpcController* controller,
              const pairec::pipeline::HealthRequest* request,
              pairec::pipeline::HealthResponse* response,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    CompleteHealth(backend_, request, response, "vector_recall_adapter", controller);
  }

 private:
  LocalBackend* backend_;
};
#else
class RankService final : public pairec::pipeline::DeepFMRankService {
 public:
  explicit RankService(LocalBackend* backend) : backend_(backend) {}

  void Rank(google::protobuf::RpcController* controller,
            const pairec::pipeline::RankRequest* request,
            pairec::pipeline::RankResponse* response,
            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    const auto started = Clock::now();
    std::string error;
    int64_t backend_rpc_us = 0;
    bool ok = ValidateContext(*request, &error);
    if (ok) {
      ok = Forward(backend_, "/rank", *request, response, &backend_rpc_us, &error);
    }
    if (!ok) {
      response->set_code(500);
      response->set_message(error);
    }
    CompleteTrace(*request, response, "deepfm_rank_adapter", started, backend_rpc_us,
                  ok ? "ok" : "error", error);
    if (!ok) {
      static_cast<brpc::Controller*>(controller)->SetFailed(error);
    }
  }

  void Health(google::protobuf::RpcController* controller,
              const pairec::pipeline::HealthRequest* request,
              pairec::pipeline::HealthResponse* response,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    CompleteHealth(backend_, request, response, "deepfm_rank_adapter", controller);
  }

 private:
  LocalBackend* backend_;
};
#endif

}  // namespace

int main(int argc, char** argv) {
  Config config;
  if (!ParseArgs(argc, argv, &config)) {
    return 2;
  }
  LocalBackend backend;
  if (!backend.Init(config)) {
    std::cerr << "Failed to initialize localhost backend " << config.backend << std::endl;
    return 1;
  }
  brpc::Server server;
#if defined(PAIREC_VECTOR_ADAPTER)
  VectorService service(&backend);
  const char* component = "vector_recall_adapter";
#else
  RankService service(&backend);
  const char* component = "deepfm_rank_adapter";
#endif
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    std::cerr << "Failed to add " << component << " service" << std::endl;
    return 1;
  }
  brpc::ServerOptions options;
  options.idle_timeout_sec = config.idle_timeout_sec;
  if (server.Start(config.listen_port, &options) != 0) {
    std::cerr << "Failed to start " << component << " on port " << config.listen_port << std::endl;
    return 1;
  }
  std::cout << component << " listening on 0.0.0.0:" << config.listen_port
            << " backend=http://" << config.backend
            << " backend_timeout_ms=" << config.backend_timeout_ms
            << " max_retry=0" << std::endl;
  server.RunUntilAskedToQuit();
  return 0;
}
