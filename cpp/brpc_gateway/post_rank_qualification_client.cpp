#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <openssl/sha.h>

#if __has_include("pipeline_service.pb.h")
#include "pipeline_service.pb.h"
#else
#include "pairec_ub_probe/pipeline_service.pb.h"
#endif

namespace {

struct Config {
  std::string server;
  int requests = 1;
  int timeout_ms = 5000;
  int payload_bytes = 102400;
  std::string request_prefix = "post-rank-qualification";
  std::string start_file;
  int start_wait_timeout_ms = 60000;
};

bool Consume(const char* argument, const std::string& name, std::string* value) {
  const std::string prefix = "--" + name + "=";
  const std::string input(argument == nullptr ? "" : argument);
  if (input.rfind(prefix, 0) != 0) return false;
  *value = input.substr(prefix.size());
  return true;
}

bool Parse(int argc, char** argv, Config* config) {
  for (int index = 1; index < argc; ++index) {
    std::string value;
    if (Consume(argv[index], "server", &value)) config->server = value;
    else if (Consume(argv[index], "requests", &value)) config->requests = std::atoi(value.c_str());
    else if (Consume(argv[index], "timeout_ms", &value)) config->timeout_ms = std::atoi(value.c_str());
    else if (Consume(argv[index], "payload_bytes", &value)) config->payload_bytes = std::atoi(value.c_str());
    else if (Consume(argv[index], "request_prefix", &value)) config->request_prefix = value;
    else if (Consume(argv[index], "start_file", &value)) config->start_file = value;
    else if (Consume(argv[index], "start_wait_timeout_ms", &value)) {
      config->start_wait_timeout_ms = std::atoi(value.c_str());
    }
    else return false;
  }
  return !config->server.empty() && config->requests > 0 &&
      config->timeout_ms > 0 && config->payload_bytes == 102400 &&
      !config->request_prefix.empty() && config->start_wait_timeout_ms > 0;
}

std::string Sha256(const std::string& value) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(), digest);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned char byte : digest) output << std::setw(2) << static_cast<unsigned>(byte);
  return output.str();
}

}  // namespace

int main(int argc, char** argv) {
  Config config;
  if (!Parse(argc, argv, &config)) {
    std::cerr << "Usage: " << argv[0]
              << " --server=HOST:PORT [--requests=N] [--timeout_ms=5000]"
              << " [--payload_bytes=102400] [--start_file=PATH]" << std::endl;
    return 2;
  }

  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "single";
  options.timeout_ms = config.timeout_ms;
  options.max_retry = 0;
  brpc::Channel channel;
  if (channel.Init(config.server.c_str(), "", &options) != 0) {
    std::cerr << "Failed to initialize qualification channel to " << config.server << std::endl;
    return 1;
  }
  pairec::pipeline::DeepFMRankService_Stub stub(&channel);

  if (!config.start_file.empty()) {
    std::cout << "POST_RANK_QUALIFICATION_READY server=" << config.server
              << " requests=" << config.requests << " payload_bytes="
              << config.payload_bytes << std::endl;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config.start_wait_timeout_ms);
    while (!std::ifstream(config.start_file).good()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::cerr << "timed out waiting for start_file=" << config.start_file << std::endl;
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "POST_RANK_QUALIFICATION_START_RELEASED start_file="
              << config.start_file << std::endl;
  }

  for (int sequence = 1; sequence <= config.requests; ++sequence) {
    const std::string request_id = config.request_prefix + "-" + std::to_string(sequence);
    pairec::pipeline::RankRequest request;
    request.mutable_context()->set_request_id(request_id);
    request.mutable_context()->set_span_id("post-rank-qualification");
    request.mutable_context()->set_contract_version("pairec.pipeline_trace.v1");
    request.set_user_id("ub-qualification-user");
    request.set_payload_padding(std::string(config.payload_bytes, 'B'));
    std::string ordered_ids;
    for (int candidate = 0; candidate < 50; ++candidate) {
      const std::string item_id = "qualification-item-" + std::to_string(candidate);
      request.add_items()->set_item_id(item_id);
      ordered_ids.append(item_id).push_back('\n');
    }

    pairec::pipeline::RankResponse response;
    brpc::Controller controller;
    controller.set_timeout_ms(config.timeout_ms);
    const auto started = std::chrono::steady_clock::now();
    stub.Rank(&controller, &request, &response, nullptr);
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (controller.Failed() || response.code() != 200 || response.items_size() != 50) {
      std::cerr << "post-rank qualification request failed request_id=" << request_id
                << " controller=" << controller.ErrorText() << " code=" << response.code()
                << " items=" << response.items_size() << std::endl;
      return 1;
    }
    std::string response_ids;
    for (int candidate = 0; candidate < response.items_size(); ++candidate) {
      if (response.items(candidate).item_id() != request.items(candidate).item_id()) {
        std::cerr << "post-rank qualification order mismatch request_id=" << request_id
                  << " index=" << candidate << std::endl;
        return 1;
      }
      response_ids.append(response.items(candidate).item_id()).push_back('\n');
    }
    const std::string request_sha = Sha256(ordered_ids);
    const std::string response_sha = Sha256(response_ids);
    if (request_sha != response_sha) {
      std::cerr << "post-rank qualification SHA mismatch request_id=" << request_id << std::endl;
      return 1;
    }
    std::cout << "{\"event\":\"post_rank_qualification_request\",\"request_id\":\""
              << request_id << "\",\"payload_bytes\":" << config.payload_bytes
              << ",\"candidate_count\":50,\"candidate_sha256\":\"" << request_sha
              << "\",\"wall_ms\":" << std::fixed << std::setprecision(3) << wall_ms
              << ",\"valid\":true}" << std::endl;
  }
  std::cout << "POST_RANK_QUALIFICATION_PASS requests=" << config.requests
            << " payload_bytes=" << config.payload_bytes << std::endl;
  return 0;
}
