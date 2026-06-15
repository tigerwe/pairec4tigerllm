#include <cstdlib>
#include <iostream>
#include <string>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/time.h>

#include "recommend.pb.h"

namespace {

struct ClientConfig {
  std::string server = "127.0.0.1:18100";
  std::string method = "recommend";
  std::string user_id = "brpc_smoke";
  int topk = 5;
  int timeout_ms = 5000;
  int max_retry = 1;
  int requests = 1;
  bool print_raw_json = false;
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
      << "  --server=127.0.0.1:18100\n"
      << "  --method=recommend|health\n"
      << "  --user_id=brpc_smoke\n"
      << "  --topk=5\n"
      << "  --requests=1\n"
      << "  --timeout_ms=5000\n"
      << "  --max_retry=1\n"
      << "  --print_raw_json=0|1\n";
}

bool ParseArgs(int argc, char** argv, ClientConfig* config) {
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (ConsumeArgValue(argv[i], "server", &value)) {
      config->server = value;
    } else if (ConsumeArgValue(argv[i], "method", &value)) {
      config->method = value;
    } else if (ConsumeArgValue(argv[i], "user_id", &value)) {
      config->user_id = value;
    } else if (ConsumeArgValue(argv[i], "topk", &value)) {
      config->topk = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "requests", &value)) {
      config->requests = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "timeout_ms", &value)) {
      config->timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "max_retry", &value)) {
      config->max_retry = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "print_raw_json", &value)) {
      config->print_raw_json = value == "1" || value == "true";
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

void AddSemanticId(
    pairec::inference::RecommendRequest* request,
    int s0,
    int s1,
    int s2,
    int s3) {
  auto* semantic_id = request->add_history();
  semantic_id->add_value(s0);
  semantic_id->add_value(s1);
  semantic_id->add_value(s2);
  semantic_id->add_value(s3);
}

pairec::inference::RecommendRequest BuildRequest(const ClientConfig& config, int index) {
  pairec::inference::RecommendRequest request;
  request.set_user_id(config.user_id + "_" + std::to_string(index));
  request.set_topk(config.topk);
  request.set_temperature(1.0);
  request.set_beam_width(1);
  request.set_request_id("brpc-smoke-" + std::to_string(index));
  AddSemanticId(&request, 169, 41, 0, 0);
  AddSemanticId(&request, 20, 53, 0, 0);
  AddSemanticId(&request, 80, 201, 0, 0);
  return request;
}

}  // namespace

int main(int argc, char** argv) {
  ClientConfig config;
  if (!ParseArgs(argc, argv, &config)) {
    return 2;
  }

  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "pooled";
  options.timeout_ms = config.timeout_ms;
  options.max_retry = config.max_retry;

  brpc::Channel channel;
  if (channel.Init(config.server.c_str(), "", &options) != 0) {
    std::cerr << "Failed to initialize brpc channel to " << config.server << std::endl;
    return 1;
  }
  pairec::inference::RecommendService_Stub stub(&channel);

  int ok_count = 0;
  butil::Timer total_timer;
  total_timer.start();
  for (int i = 1; i <= config.requests; ++i) {
    brpc::Controller cntl;
    butil::Timer timer;
    timer.start();

    if (config.method == "health") {
      pairec::inference::HealthRequest request;
      pairec::inference::HealthResponse response;
      stub.Health(&cntl, &request, &response, nullptr);
      timer.stop();
      if (cntl.Failed()) {
        std::cerr << "health failed latency_ms=" << timer.m_elapsed()
                  << " error=" << cntl.ErrorText() << std::endl;
        continue;
      }
      ++ok_count;
      std::cout << "health ok latency_ms=" << timer.m_elapsed()
                << " code=" << response.code()
                << " status=" << response.status() << std::endl;
      if (config.print_raw_json) {
        std::cout << response.raw_json() << std::endl;
      }
    } else {
      auto request = BuildRequest(config, i);
      pairec::inference::RecommendResponse response;
      stub.Recommend(&cntl, &request, &response, nullptr);
      timer.stop();
      if (cntl.Failed()) {
        std::cerr << "recommend failed index=" << i
                  << " latency_ms=" << timer.m_elapsed()
                  << " error=" << cntl.ErrorText() << std::endl;
        continue;
      }
      ++ok_count;
      std::cout << "recommend ok index=" << i
                << " latency_ms=" << timer.m_elapsed()
                << " code=" << response.code()
                << " user_id=" << response.user_id()
                << " items=" << response.recommendations_size()
                << " inference_ms=" << response.inference_time_ms()
                << std::endl;
      if (config.print_raw_json) {
        std::cout << response.raw_json() << std::endl;
      }
    }
  }
  total_timer.stop();
  std::cout << "summary ok=" << ok_count
            << " total=" << config.requests
            << " total_ms=" << total_timer.m_elapsed() << std::endl;
  return ok_count == config.requests ? 0 : 1;
}
