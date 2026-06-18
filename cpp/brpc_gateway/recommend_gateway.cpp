#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/time.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/util/json_util.h>

#include "recommend.pb.h"

namespace {

struct GatewayConfig {
  int listen_port = 18100;
  std::string http_forward = "127.0.0.1:18000";
  int http_timeout_ms = 5000;
  int http_max_retry = 0;
  int idle_timeout_sec = -1;
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
      << "  --listen_port=18100\n"
      << "  --http_forward=127.0.0.1:18000\n"
      << "  --http_timeout_ms=5000\n"
      << "  --http_max_retry=0\n"
      << "  --idle_timeout_sec=-1\n";
}

bool ParseArgs(int argc, char** argv, GatewayConfig* config) {
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (ConsumeArgValue(argv[i], "listen_port", &value)) {
      config->listen_port = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "http_forward", &value)) {
      config->http_forward = value;
    } else if (ConsumeArgValue(argv[i], "http_timeout_ms", &value)) {
      config->http_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "http_max_retry", &value)) {
      config->http_max_retry = std::atoi(value.c_str());
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
  return true;
}

std::string JsonEscape(const std::string& input) {
  std::string output;
  output.reserve(input.size() + 8);
  for (char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
          output += buf;
        } else {
          output += ch;
        }
    }
  }
  return output;
}

std::string BuildRecommendJson(const pairec::inference::RecommendRequest& request) {
  std::ostringstream out;
  out << "{\"user_id\":\"" << JsonEscape(request.user_id()) << "\",";
  out << "\"history\":[";
  for (int i = 0; i < request.history_size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "[";
    const auto& values = request.history(i).value();
    for (int j = 0; j < values.size(); ++j) {
      if (j != 0) {
        out << ",";
      }
      out << values.Get(j);
    }
    out << "]";
  }
  out << "],";
  out << "\"topk\":" << request.topk() << ",";
  out << "\"temperature\":" << request.temperature() << ",";
  out << "\"beam_width\":" << request.beam_width();
  out << "}";
  return out.str();
}

class HttpForwarder {
 public:
  bool Init(const GatewayConfig& config) {
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.connection_type = "pooled";
    options.timeout_ms = config.http_timeout_ms;
    options.max_retry = config.http_max_retry;
    return channel_.Init(config.http_forward.c_str(), "", &options) == 0;
  }

  bool PostJson(
      const std::string& path,
      const std::string& body,
      const std::string& request_id,
      std::string* response_body,
      int* status_code,
      std::string* error) {
    brpc::Controller cntl;
    cntl.http_request().uri() = path;
    cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
    cntl.http_request().SetHeader("Content-Type", "application/json");
    if (!request_id.empty()) {
      cntl.http_request().SetHeader("X-Request-ID", request_id);
    }
    cntl.request_attachment().append(body);

    channel_.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
    if (cntl.Failed()) {
      *status_code = 0;
      *error = cntl.ErrorText();
      return false;
    }
    *status_code = cntl.http_response().status_code();
    cntl.response_attachment().copy_to(response_body);
    return *status_code >= 200 && *status_code < 300;
  }

  bool Get(
      const std::string& path,
      std::string* response_body,
      int* status_code,
      std::string* error) {
    brpc::Controller cntl;
    cntl.http_request().uri() = path;
    cntl.http_request().set_method(brpc::HTTP_METHOD_GET);

    channel_.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
    if (cntl.Failed()) {
      *status_code = 0;
      *error = cntl.ErrorText();
      return false;
    }
    *status_code = cntl.http_response().status_code();
    cntl.response_attachment().copy_to(response_body);
    return *status_code >= 200 && *status_code < 300;
  }

 private:
  brpc::Channel channel_;
};

class RecommendServiceImpl final : public pairec::inference::RecommendService {
 public:
  explicit RecommendServiceImpl(HttpForwarder* forwarder) : forwarder_(forwarder) {}

  void Recommend(
      google::protobuf::RpcController* controller,
      const pairec::inference::RecommendRequest* request,
      pairec::inference::RecommendResponse* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);
    butil::Timer timer;
    timer.start();

    std::string body;
    int status_code = 0;
    std::string error;
    const std::string json = BuildRecommendJson(*request);
    const bool ok = forwarder_->PostJson(
        "/recommend", json, request->request_id(), &body, &status_code, &error);

    response->set_raw_json(body);
    if (!ok) {
      response->set_code(status_code == 0 ? 500 : status_code);
      response->set_user_id(request->user_id());
      response->set_error(error.empty() ? "HTTP forward failed" : error);
      cntl->SetFailed(response->error());
      return;
    }

    google::protobuf::util::JsonParseOptions parse_options;
    parse_options.ignore_unknown_fields = true;
    const auto parse_status =
        google::protobuf::util::JsonStringToMessage(body, response, parse_options);
    response->set_raw_json(body);
    if (!parse_status.ok()) {
      response->set_code(status_code);
      response->set_user_id(request->user_id());
      response->set_error("HTTP response JSON parse failed: " + parse_status.ToString());
      return;
    }
    if (!response->has_code()) {
      response->set_code(status_code);
    }
    if (!response->has_user_id()) {
      response->set_user_id(request->user_id());
    }
    timer.stop();
    std::cout << "[brpc-gateway] method=Recommend user=" << request->user_id()
              << " code=" << response->code()
              << " items=" << response->recommendations_size()
              << " latency_ms=" << timer.m_elapsed()
              << " http_status=" << status_code << std::endl;
  }

  void Health(
      google::protobuf::RpcController* controller,
      const pairec::inference::HealthRequest*,
      pairec::inference::HealthResponse* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);

    std::string body;
    int status_code = 0;
    std::string error;
    const bool ok = forwarder_->Get("/health", &body, &status_code, &error);
    response->set_code(status_code == 0 ? 500 : status_code);
    response->set_raw_json(body);
    if (!ok) {
      response->set_status(error.empty() ? "unhealthy" : error);
      cntl->SetFailed(response->status());
      return;
    }

    google::protobuf::util::JsonParseOptions parse_options;
    parse_options.ignore_unknown_fields = true;
    const auto parse_status =
        google::protobuf::util::JsonStringToMessage(body, response, parse_options);
    response->set_raw_json(body);
    if (!parse_status.ok() && !response->has_status()) {
      response->set_status("healthy");
    }
    if (!response->has_code()) {
      response->set_code(status_code);
    }
  }

 private:
  HttpForwarder* forwarder_;
};

}  // namespace

int main(int argc, char** argv) {
  GatewayConfig config;
  if (!ParseArgs(argc, argv, &config)) {
    return 2;
  }

  HttpForwarder forwarder;
  if (!forwarder.Init(config)) {
    std::cerr << "Failed to initialize HTTP forward channel to "
              << config.http_forward << std::endl;
    return 1;
  }

  RecommendServiceImpl service(&forwarder);
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    std::cerr << "Failed to add RecommendService" << std::endl;
    return 1;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = config.idle_timeout_sec;
  if (server.Start(config.listen_port, &options) != 0) {
    std::cerr << "Failed to start brpc gateway on port "
              << config.listen_port << std::endl;
    return 1;
  }

  std::cout << "brpc gateway listening on 0.0.0.0:" << config.listen_port
            << ", forwarding to http://" << config.http_forward << std::endl;
  server.RunUntilAskedToQuit();
  return 0;
}
