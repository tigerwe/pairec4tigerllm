#include <cstdlib>
#include <iostream>
#include <string>

#include <brpc/channel.h>
#include <brpc/controller.h>

#include "pipeline_service.pb.h"

namespace {

struct Config {
  std::string server;
  std::string service;
  int timeout_ms = 1000;
  std::string control;
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
    if (Consume(argv[index], "server", &value)) {
      config->server = value;
    } else if (Consume(argv[index], "service", &value)) {
      config->service = value;
    } else if (Consume(argv[index], "timeout_ms", &value)) {
      config->timeout_ms = std::atoi(value.c_str());
    } else if (Consume(argv[index], "control", &value)) {
      config->control = value;
    } else {
      return false;
    }
  }
  const bool control_valid = config->control.empty() || config->control == "arm" ||
      config->control == "disarm" || config->control == "status";
  return !config->server.empty() && control_valid &&
         (config->service == "vector" || config->service == "rank") &&
         config->timeout_ms > 0;
}

}  // namespace

int main(int argc, char** argv) {
  Config config;
  if (!ParseArgs(argc, argv, &config)) {
    std::cerr << "Usage: " << argv[0]
              << " --server=host:port --service=vector|rank [--timeout_ms=1000]"
              << " [--control=arm|disarm|status]"
              << std::endl;
    return 2;
  }
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "single";
  options.timeout_ms = config.timeout_ms;
  options.max_retry = 0;
  brpc::Channel channel;
  if (channel.Init(config.server.c_str(), "", &options) != 0) {
    std::cerr << "channel init failed" << std::endl;
    return 1;
  }
  pairec::pipeline::HealthRequest request;
  request.mutable_context()->set_request_id("pipeline-health");
  request.mutable_context()->set_span_id("health");
  request.mutable_context()->set_contract_version("pairec.pipeline_trace.v1");
  if (!config.control.empty()) {
    request.set_payload_padding("PAIREC_RETURN_CONTROL_V1:" + config.control);
  }
  pairec::pipeline::HealthResponse response;
  brpc::Controller controller;
  if (config.service == "vector") {
    pairec::pipeline::VectorRecallService_Stub stub(&channel);
    stub.Health(&controller, &request, &response, nullptr);
  } else {
    pairec::pipeline::DeepFMRankService_Stub stub(&channel);
    stub.Health(&controller, &request, &response, nullptr);
  }
  const std::string expected_status = config.control.empty() ? "healthy" :
      (config.control == "disarm" ? "disarmed" :
       (config.control == "arm" ? "armed" : response.status()));
  if (controller.Failed() || response.code() != 200 ||
      response.status() != expected_status ||
      !response.has_trace() || response.trace().context().request_id() != "pipeline-health") {
    std::cerr << "health failed: " << controller.ErrorText()
              << " code=" << response.code() << " status=" << response.status() << std::endl;
    return 1;
  }
  std::cout << "health ok service=" << config.service
            << " backend=" << response.backend()
            << " total_us=" << response.trace().total_us()
            << " control=" << (config.control.empty() ? "none" : config.control)
            << " status=" << response.status() << std::endl;
  return 0;
}
