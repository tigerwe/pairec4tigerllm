#include "kvc_burst_shared.h"

#include <datasystem/kv_client.h>

#include <fcntl.h>
#include <sys/mman.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

namespace
{

using pairec::kvc_burst::SharedControl;
using pairec::kvc_burst::State;

struct Config
{
    std::string host{"127.0.0.1"};
    int port{18482};
    uint64_t objectSize{1792ULL * 1024ULL};
    std::string prefix{"PairecKvcBurst"};
    std::string requestId{"kvc-burst-probe"};
    std::string controlPath{"/run/pairec-kvc-burst/control"};
    uint32_t readyTimeoutMs{30000};
    uint32_t resultTimeoutMs{120000};
    bool waitResult{true};
    bool cleanupKey{true};
};

struct Mapping
{
    int fd{-1};
    SharedControl* control{nullptr};
    ~Mapping()
    {
        if (control != nullptr) ::munmap(control, sizeof(SharedControl));
        if (fd >= 0) ::close(fd);
    }
};

bool ParseUnsigned(const std::string& value, uint64_t* output)
{
    try
    {
        size_t used = 0;
        auto parsed = std::stoull(value, &used);
        if (used != value.size()) return false;
        *output = parsed;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool ParseBool(const std::string& value, bool* output)
{
    if (value == "1" || value == "true")
    {
        *output = true;
        return true;
    }
    if (value == "0" || value == "false")
    {
        *output = false;
        return true;
    }
    return false;
}

bool ParseArgs(int argc, char** argv, Config* config)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string argument(argv[i]);
        auto split = argument.find('=');
        if (argument.rfind("--", 0) != 0 || split == std::string::npos)
        {
            std::cerr << "invalid argument: " << argument << std::endl;
            return false;
        }
        auto name = argument.substr(2, split - 2);
        auto value = argument.substr(split + 1);
        uint64_t parsed = 0;
        if (name == "host") config->host = value;
        else if (name == "port")
        {
            if (!ParseUnsigned(value, &parsed)) return false;
            config->port = static_cast<int>(parsed);
        }
        else if (name == "object_size")
        {
            if (!ParseUnsigned(value, &config->objectSize)) return false;
        }
        else if (name == "prefix") config->prefix = value;
        else if (name == "request_id") config->requestId = value;
        else if (name == "control_path") config->controlPath = value;
        else if (name == "ready_timeout_ms")
        {
            if (!ParseUnsigned(value, &parsed)) return false;
            config->readyTimeoutMs = static_cast<uint32_t>(parsed);
        }
        else if (name == "result_timeout_ms")
        {
            if (!ParseUnsigned(value, &parsed)) return false;
            config->resultTimeoutMs = static_cast<uint32_t>(parsed);
        }
        else if (name == "wait_result")
        {
            if (!ParseBool(value, &config->waitResult)) return false;
        }
        else if (name == "cleanup_key")
        {
            if (!ParseBool(value, &config->cleanupKey)) return false;
        }
        else
        {
            std::cerr << "unknown argument: " << name << std::endl;
            return false;
        }
    }
    return !config->host.empty() && config->port > 0 && config->port <= 65535 && config->objectSize > 0
        && !config->prefix.empty() && !config->requestId.empty() && !config->controlPath.empty()
        && config->readyTimeoutMs > 0 && config->resultTimeoutMs > 0;
}

std::unique_ptr<datasystem::KVClient> CreateClient(const Config& config)
{
    datasystem::ConnectOptions options;
    options.host = config.host;
    options.port = config.port;
    options.enableCrossNodeConnection = true;
    options.enableExclusiveConnection = true;
    auto client = std::make_unique<datasystem::KVClient>(options);
    auto status = client->Init();
    if (status.IsError())
    {
        std::cerr << "business KVClient Init failed: " << status.ToString() << std::endl;
        return nullptr;
    }
    return client;
}

bool OpenMapping(const Config& config, Mapping* mapping)
{
    auto deadline = pairec::kvc_burst::DeadlineNs(config.readyTimeoutMs);
    while (pairec::kvc_burst::MonotonicNs() < deadline)
    {
        mapping->fd = ::open(config.controlPath.c_str(), O_RDWR);
        if (mapping->fd >= 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (mapping->fd < 0)
    {
        std::perror("open control path");
        return false;
    }
    auto address = ::mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
    if (address == MAP_FAILED)
    {
        std::perror("mmap control path");
        return false;
    }
    mapping->control = static_cast<SharedControl*>(address);
    return pairec::kvc_burst::IsCompatible(*mapping->control);
}

bool WaitReady(SharedControl& control, uint32_t timeoutMs)
{
    auto deadline = pairec::kvc_burst::DeadlineNs(timeoutMs);
    while (pairec::kvc_burst::MonotonicNs() < deadline)
    {
        if (pairec::kvc_burst::Load(&control.state) == static_cast<uint32_t>(State::kReady)
            && pairec::kvc_burst::Load(&control.armed_pressure_lanes)
                == pairec::kvc_burst::Load(&control.pressure_lanes))
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

void DeleteKey(datasystem::KVClient& client, const std::string& key)
{
    std::vector<std::string> failed;
    auto status = client.Del(std::vector<std::string>{key}, failed);
    if (status.IsError() || !failed.empty())
    {
        std::cerr << "business key cleanup failed: " << status.ToString() << std::endl;
    }
}

} // namespace

int main(int argc, char** argv)
{
    Config config;
    if (!ParseArgs(argc, argv, &config)) return 2;

    Mapping mapping;
    if (!OpenMapping(config, &mapping))
    {
        std::cerr << "KVC burst control is missing or incompatible" << std::endl;
        return 1;
    }
    auto& control = *mapping.control;
    if (pairec::kvc_burst::Load(&control.object_size_bytes) != config.objectSize)
    {
        std::cerr << "object size mismatch wrapper=" << control.object_size_bytes << " probe=" << config.objectSize
                  << std::endl;
        return 1;
    }

    auto client = CreateClient(config);
    if (!client) return 1;
    auto businessKey = config.prefix + "_business";
    std::string value(config.objectSize, 'b');
    datasystem::SetParam param;
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    auto setStatus = client->Set(businessKey, datasystem::StringView(value), param);
    if (setStatus.IsError())
    {
        std::cerr << "business key prefill failed: " << setStatus.ToString() << std::endl;
        return 1;
    }
    datasystem::Optional<datasystem::Buffer> verification;
    auto verifyStatus = client->Get(businessKey, verification, 0);
    if (verifyStatus.IsError() || !verification)
    {
        std::cerr << "business key verification failed: " << verifyStatus.ToString() << std::endl;
        return 1;
    }

    if (!WaitReady(control, config.readyTimeoutMs))
    {
        std::cerr << "KVC burst wrapper did not become ready" << std::endl;
        return 1;
    }
    uint32_t expected = static_cast<uint32_t>(State::kReady);
    pairec::kvc_burst::Store(&control.claim_started_ns, pairec::kvc_burst::MonotonicNs());
    if (!pairec::kvc_burst::CompareExchange(
            &control.state, &expected, static_cast<uint32_t>(State::kClaimed)))
    {
        std::cerr << "KVC burst wrapper is busy state=" << expected << std::endl;
        return 1;
    }

    auto generation = pairec::kvc_burst::Load(&control.prepared_generation);
    pairec::kvc_burst::CopyRequestId(control.request_id, config.requestId.c_str());
    pairec::kvc_burst::Store(&control.state, static_cast<uint32_t>(State::kRunning));
    pairec::kvc_burst::Store(&control.run_generation, generation);
    pairec::kvc_burst::FutexWake(&control.run_generation);

    uint64_t barrierWaitUs = 0;
    auto released = pairec::kvc_burst::ArriveAndWait(
        &control, generation, pairec::kvc_burst::Load(&control.barrier_timeout_ms), &barrierWaitUs);
    pairec::kvc_burst::Store(&control.business_barrier_wait_us, barrierWaitUs);

    auto businessStart = pairec::kvc_burst::MonotonicNs();
    pairec::kvc_burst::Store(&control.business_start_ns, businessStart);
    datasystem::Optional<datasystem::Buffer> buffer;
    auto getStatus = client->Get(businessKey, buffer, 0);
    auto businessEnd = pairec::kvc_burst::MonotonicNs();
    auto success = !getStatus.IsError() && static_cast<bool>(buffer);
    pairec::kvc_burst::Store(&control.business_end_ns, businessEnd);
    pairec::kvc_burst::Store(&control.business_success, success ? 1U : 0U);
    pairec::kvc_burst::Store(&control.business_done_generation, generation);
    pairec::kvc_burst::FutexWake(&control.business_done_generation);

    std::cout << "{\"event\":\"kvc_burst_business_complete\",\"generation\":" << generation
              << ",\"request_id\":\"" << config.requestId << "\",\"barrier_released\":"
              << (released ? "true" : "false") << ",\"barrier_wait_us\":" << barrierWaitUs
              << ",\"business_get_us\":" << (businessEnd - businessStart) / 1000ULL
              << ",\"business_success\":" << (success ? "true" : "false") << "}" << std::endl;

    bool resultValid = released && success;
    if (config.waitResult)
    {
        auto deadline = pairec::kvc_burst::DeadlineNs(config.resultTimeoutMs);
        if (!pairec::kvc_burst::WaitForValue(&control.result_generation, generation, deadline))
        {
            std::cerr << "timed out waiting for asynchronous pressure result" << std::endl;
            resultValid = false;
        }
        else
        {
            resultValid = pairec::kvc_burst::Load(&control.result_valid) != 0;
            std::cout << "{\"event\":\"kvc_burst_probe_result\",\"generation\":" << generation
                      << ",\"request_id\":\"" << config.requestId << "\",\"valid\":"
                      << (resultValid ? "true" : "false") << ",\"failure\":"
                      << pairec::kvc_burst::Load(&control.result_failure)
                      << ",\"max_active_all_gets\":"
                      << pairec::kvc_burst::Load(&control.result_max_active_all)
                      << ",\"business_overlap_gets\":"
                      << pairec::kvc_burst::Load(&control.result_business_overlap)
                      << ",\"start_skew_us\":" << pairec::kvc_burst::Load(&control.result_start_skew_us)
                      << "}" << std::endl;
        }
    }

    if (config.cleanupKey) DeleteKey(*client, businessKey);
    return resultValid ? 0 : 1;
}
