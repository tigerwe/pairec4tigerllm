#include "payload_integrity.h"
#include "ubsocket_trace_key_workaround.h"

#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/time.h>
#include <gflags/gflags.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "pairec_ub_probe/recommend.pb.h"

DEFINE_int32(probe_port, 18100, "Port for the minimal RecommendService server");
DEFINE_int32(probe_idle_timeout_sec, -1, "Server connection idle timeout");
DEFINE_uint64(probe_max_payload_bytes, 4ULL * 1024ULL * 1024ULL, "Maximum accepted payload size");
DEFINE_bool(probe_echo_payload, true, "Echo payload through the bRPC response attachment");
DEFINE_bool(ubsocket_use_ub, false, "Use UBSocket/UB for this bRPC server");

namespace
{

class MinimalRecommendService final : public pairec::inference::RecommendService
{
public:
    void Recommend(
        google::protobuf::RpcController* controller,
        const pairec::inference::RecommendRequest* request,
        pairec::inference::RecommendResponse* response,
        google::protobuf::Closure* done) override
    {
        brpc::ClosureGuard doneGuard(done);
        auto* brpcController = static_cast<brpc::Controller*>(controller);
        butil::Timer timer;
        timer.start();

        const std::string& payload = request->payload_padding();
        if (payload.size() > FLAGS_probe_max_payload_bytes)
        {
            response->set_code(413);
            response->set_user_id(request->user_id());
            response->set_error("payload exceeds probe_max_payload_bytes");
            return;
        }

        const std::string sha256 = pairec::brpc_ub_probe::Sha256Hex(payload);
        response->set_code(200);
        response->set_user_id(request->user_id());
        response->set_raw_json(
            pairec::brpc_ub_probe::BuildPayloadMetadata("Recommend", payload.size(), sha256));
        auto* recommendation = response->add_recommendations();
        recommendation->set_item_id(1);
        recommendation->set_score(1.0);
        if (FLAGS_probe_echo_payload)
        {
            brpcController->response_attachment().append(payload);
        }
        timer.stop();
        response->set_inference_time_ms(timer.m_elapsed());

        std::cout << "{\"event\":\"minimal_recommend_request\",\"method\":\"Recommend\""
                  << ",\"request_id\":\"" << request->request_id() << "\""
                  << ",\"payload_bytes\":" << payload.size()
                  << ",\"payload_sha256\":\"" << sha256 << "\""
                  << ",\"latency_ms\":" << timer.m_elapsed() << "}" << std::endl;
    }

    void Health(
        google::protobuf::RpcController* controller,
        const pairec::inference::HealthRequest* request,
        pairec::inference::HealthResponse* response,
        google::protobuf::Closure* done) override
    {
        brpc::ClosureGuard doneGuard(done);
        auto* brpcController = static_cast<brpc::Controller*>(controller);
        const std::string& payload = request->payload_padding();
        if (payload.size() > FLAGS_probe_max_payload_bytes)
        {
            response->set_code(413);
            response->set_status("payload too large");
            return;
        }

        const std::string sha256 = pairec::brpc_ub_probe::Sha256Hex(payload);
        response->set_code(200);
        response->set_status("healthy");
        response->set_backend("minimal_brpc_ub_probe");
        response->set_raw_json(
            pairec::brpc_ub_probe::BuildPayloadMetadata("Health", payload.size(), sha256));
        if (FLAGS_probe_echo_payload)
        {
            brpcController->response_attachment().append(payload);
        }

        std::cout << "{\"event\":\"minimal_recommend_request\",\"method\":\"Health\""
                  << ",\"payload_bytes\":" << payload.size()
                  << ",\"payload_sha256\":\"" << sha256 << "\"}" << std::endl;
    }
};

} // namespace

int main(int argc, char** argv)
{
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_ubsocket_use_ub && !pairec::brpc_ub_probe::InitializeUBSocketTraceKeys())
    {
        return 1;
    }

    MinimalRecommendService service;
    brpc::Server server;
    if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0)
    {
        std::cerr << "Failed to add RecommendService" << std::endl;
        return 1;
    }

    brpc::ServerOptions options;
    options.idle_timeout_sec = FLAGS_probe_idle_timeout_sec;
    options.use_ub = FLAGS_ubsocket_use_ub;
    if (server.Start(FLAGS_probe_port, &options) != 0)
    {
        std::cerr << "Failed to start minimal RecommendService on port " << FLAGS_probe_port << std::endl;
        return 1;
    }

    std::cout << "MINIMAL_RECOMMEND_UB_SERVER_READY port=" << FLAGS_probe_port
              << " max_payload_bytes=" << FLAGS_probe_max_payload_bytes
              << " echo_payload=" << (FLAGS_probe_echo_payload ? "true" : "false") << std::endl;
    server.RunUntilAskedToQuit();
    return 0;
}
