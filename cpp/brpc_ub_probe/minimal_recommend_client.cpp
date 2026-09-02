#include "payload_integrity.h"

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/time.h>
#include <gflags/gflags.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pairec_ub_probe/recommend.pb.h"

DEFINE_string(probe_server, "127.0.0.1:18100", "Minimal RecommendService endpoint");
DEFINE_string(probe_method, "recommend", "RPC method: recommend or health");
DEFINE_string(
    probe_payload_sizes,
    "0,1,4096,4097,65536,1048576,3670016",
    "Comma-separated payload sizes in bytes");
DEFINE_uint64(probe_seed, 20260902ULL, "Base seed for deterministic payload generation");
DEFINE_int32(probe_requests_per_size, 1, "Requests sent for each payload size");
DEFINE_int32(probe_timeout_ms, 15000, "RPC timeout in milliseconds");
DEFINE_int32(probe_connect_timeout_ms, 20000, "UB connection timeout in milliseconds");
DEFINE_int32(probe_max_retry, 0, "bRPC retry count");
DEFINE_string(
    probe_connection_type,
    "",
    "bRPC connection type; empty uses the protocol default, matching echo_c++_client");
DEFINE_bool(probe_expect_echo, true, "Require an exact payload in the response attachment");
DEFINE_string(probe_user_id, "minimal-brpc-ub-probe", "Recommend request user ID prefix");
DEFINE_bool(ubsocket_use_ub, false, "Use UBSocket/UB for this bRPC channel");

namespace
{

std::vector<size_t> ParsePayloadSizes(const std::string& text)
{
    std::vector<size_t> sizes;
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, ','))
    {
        if (token.empty())
        {
            throw std::invalid_argument("empty payload size");
        }
        size_t used = 0;
        const unsigned long long value = std::stoull(token, &used);
        if (used != token.size())
        {
            throw std::invalid_argument("invalid payload size: " + token);
        }
        sizes.push_back(static_cast<size_t>(value));
    }
    if (sizes.empty())
    {
        throw std::invalid_argument("probe_payload_sizes is empty");
    }
    return sizes;
}

bool VerifyResponse(
    brpc::Controller& controller,
    const std::string& payload,
    const std::string& actualMetadata,
    const std::string& expectedMetadata,
    std::string* error)
{
    if (actualMetadata != expectedMetadata)
    {
        *error = "server payload metadata mismatch";
        return false;
    }
    std::string echoedPayload;
    controller.response_attachment().copy_to(&echoedPayload);
    if (FLAGS_probe_expect_echo && echoedPayload != payload)
    {
        *error = "response attachment payload mismatch";
        return false;
    }
    if (!FLAGS_probe_expect_echo && !echoedPayload.empty())
    {
        *error = "unexpected response attachment payload";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_probe_method != "recommend" && FLAGS_probe_method != "health")
    {
        std::cerr << "probe_method must be recommend or health" << std::endl;
        return 2;
    }
    if (FLAGS_probe_requests_per_size <= 0)
    {
        std::cerr << "probe_requests_per_size must be positive" << std::endl;
        return 2;
    }

    std::vector<size_t> sizes;
    try
    {
        sizes = ParsePayloadSizes(FLAGS_probe_payload_sizes);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 2;
    }

    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.connection_type = FLAGS_probe_connection_type;
    options.timeout_ms = FLAGS_probe_timeout_ms;
    options.connect_timeout_ms = FLAGS_probe_connect_timeout_ms;
    options.max_retry = FLAGS_probe_max_retry;
    options.use_ub = FLAGS_ubsocket_use_ub;

    brpc::Channel channel;
    if (channel.Init(FLAGS_probe_server.c_str(), "", &options) != 0)
    {
        std::cerr << "Failed to initialize bRPC channel to " << FLAGS_probe_server << std::endl;
        return 1;
    }
    pairec::inference::RecommendService_Stub stub(&channel);

    int total = 0;
    int passed = 0;
    for (size_t size : sizes)
    {
        for (int iteration = 0; iteration < FLAGS_probe_requests_per_size; ++iteration)
        {
            ++total;
            const uint64_t seed = FLAGS_probe_seed + static_cast<uint64_t>(size) * 131ULL
                + static_cast<uint64_t>(iteration);
            const std::string payload = pairec::brpc_ub_probe::BuildDeterministicPayload(size, seed);
            const std::string sha256 = pairec::brpc_ub_probe::Sha256Hex(payload);
            const std::string methodName = FLAGS_probe_method == "recommend" ? "Recommend" : "Health";
            const std::string expectedMetadata =
                pairec::brpc_ub_probe::BuildPayloadMetadata(methodName, payload.size(), sha256);

            brpc::Controller controller;
            butil::Timer timer;
            timer.start();
            std::string actualMetadata;
            int responseCode = 0;
            if (FLAGS_probe_method == "recommend")
            {
                pairec::inference::RecommendRequest request;
                request.set_user_id(FLAGS_probe_user_id + "-" + std::to_string(total));
                request.set_request_id("minimal-brpc-ub-" + std::to_string(total));
                request.set_topk(1);
                request.set_payload_padding(payload);
                pairec::inference::RecommendResponse response;
                stub.Recommend(&controller, &request, &response, nullptr);
                responseCode = response.code();
                actualMetadata = response.raw_json();
            }
            else
            {
                pairec::inference::HealthRequest request;
                request.set_payload_padding(payload);
                pairec::inference::HealthResponse response;
                stub.Health(&controller, &request, &response, nullptr);
                responseCode = response.code();
                actualMetadata = response.raw_json();
            }
            timer.stop();

            std::string error;
            bool valid = !controller.Failed() && responseCode == 200
                && VerifyResponse(controller, payload, actualMetadata, expectedMetadata, &error);
            if (controller.Failed())
            {
                error = controller.ErrorText();
            }
            else if (responseCode != 200)
            {
                error = "response code " + std::to_string(responseCode);
            }
            if (valid)
            {
                ++passed;
            }

            std::cout << "{\"event\":\"minimal_recommend_ub_probe\",\"method\":\"" << methodName
                      << "\",\"payload_bytes\":" << payload.size()
                      << ",\"payload_sha256\":\"" << sha256 << "\""
                      << ",\"latency_ms\":" << timer.m_elapsed()
                      << ",\"valid\":" << (valid ? "true" : "false");
            if (!error.empty())
            {
                std::cout << ",\"error\":\"" << error << "\"";
            }
            std::cout << "}" << std::endl;
        }
    }

    std::cout << "MINIMAL_RECOMMEND_UB_SUMMARY passed=" << passed << " total=" << total << std::endl;
    if (passed != total)
    {
        return 1;
    }
    std::cout << "MINIMAL_RECOMMEND_UB_MATRIX_PASS" << std::endl;
    return 0;
}
