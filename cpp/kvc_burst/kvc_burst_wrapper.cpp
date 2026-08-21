#include "kvc_burst_shared.h"

#include <datasystem/kv_client.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using pairec::kvc_burst::Failure;
using pairec::kvc_burst::RefreshState;
using pairec::kvc_burst::SharedControl;
using pairec::kvc_burst::State;

std::atomic<bool> gStop{false};
std::mutex gPressureErrorMutex;
const char* gStartupStage{"process_start"};

struct Config
{
    std::string host{"127.0.0.1"};
    int port{18482};
    uint32_t concurrency{100};
    uint32_t pressureKeyCount{0};
    uint64_t objectSize{3670016ULL};
    uint32_t barrierTimeoutMs{10};
    uint32_t pressureLeadUs{1000};
    uint64_t seed{20260804};
    std::string prefix{"PairecKvcBurstV2"};
    std::string controlPath{"/run/pairec-kvc-burst/control"};
    std::string readyFile{"/run/pairec-kvc-burst/ready"};
    std::string statsFile;
    bool cleanupKeys{true};
    bool initiallyArmed{true};
    std::string controlAction;
    bool sustainedPressure{false};
    uint32_t sustainedMaxDurationMs{1000};
    uint32_t sustainedMaxLoops{100};
    bool inProcessPressure{false};
};

struct Mapping
{
    int fd{-1};
    SharedControl* control{nullptr};

    ~Mapping()
    {
        if (control != nullptr)
        {
            ::munmap(control, sizeof(SharedControl));
        }
        if (fd >= 0)
        {
            ::close(fd);
        }
    }
};

void HandleSignal(int)
{
    gStop.store(true, std::memory_order_relaxed);
}

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
        else if (name == "concurrency")
        {
            if (!ParseUnsigned(value, &parsed)) return false;
            config->concurrency = static_cast<uint32_t>(parsed);
        }
        else if (name == "pressure_key_count")
        {
            if (!ParseUnsigned(value, &parsed)) return false;
            config->pressureKeyCount = static_cast<uint32_t>(parsed);
        }
        else if (name == "object_size")
        {
            if (!ParseUnsigned(value, &config->objectSize)) return false;
        }
        else if (name == "barrier_timeout_ms")
        {
            if (!ParseUnsigned(value, &parsed)) return false;
            config->barrierTimeoutMs = static_cast<uint32_t>(parsed);
        }
        else if (name == "pressure_lead_us")
        {
            if (!ParseUnsigned(value, &parsed) || parsed > 1000000U) return false;
            config->pressureLeadUs = static_cast<uint32_t>(parsed);
        }
        else if (name == "seed")
        {
            if (!ParseUnsigned(value, &config->seed)) return false;
        }
        else if (name == "prefix") config->prefix = value;
        else if (name == "control_path") config->controlPath = value;
        else if (name == "ready_file") config->readyFile = value;
        else if (name == "stats_file") config->statsFile = value;
        else if (name == "control_action") config->controlAction = value;
        else if (name == "sustained_pressure")
        {
            if (!ParseBool(value, &config->sustainedPressure)) return false;
        }
        else if (name == "sustained_max_duration_ms")
        {
            if (!ParseUnsigned(value, &parsed) || parsed > UINT32_MAX) return false;
            config->sustainedMaxDurationMs = static_cast<uint32_t>(parsed);
        }
        else if (name == "sustained_max_loops")
        {
            if (!ParseUnsigned(value, &parsed) || parsed > UINT32_MAX) return false;
            config->sustainedMaxLoops = static_cast<uint32_t>(parsed);
        }
        else if (name == "initially_armed")
        {
            if (!ParseBool(value, &config->initiallyArmed)) return false;
        }
        else if (name == "inprocess_pressure")
        {
            if (!ParseBool(value, &config->inProcessPressure)) return false;
        }
        else if (name == "cleanup_keys")
        {
            if (!ParseBool(value, &config->cleanupKeys)) return false;
        }
        else
        {
            std::cerr << "unknown argument: " << name << std::endl;
            return false;
        }
    }
    if (config->host.empty() || config->port <= 0 || config->port > 65535 || config->concurrency == 0
        || config->concurrency > pairec::kvc_burst::kMaxConcurrency || config->objectSize == 0
        || config->barrierTimeoutMs == 0 || config->controlPath.empty()
        || config->pressureKeyCount > config->concurrency - 1)
    {
        std::cerr << "invalid host, port, concurrency, object size, timeout, or control path" << std::endl;
        return false;
    }
    if (config->sustainedPressure
        && (config->sustainedMaxDurationMs == 0 || config->sustainedMaxLoops == 0))
    {
        std::cerr << "sustained pressure requires positive max duration and max loops" << std::endl;
        return false;
    }
    if (config->inProcessPressure
        && (config->concurrency != 32U || config->pressureKeyCount == 0U
            || config->pressureKeyCount > 31U
            || config->objectSize != 3670016ULL || config->sustainedPressure
            || config->initiallyArmed))
    {
        std::cerr << "in-process pressure requires c32, 1..31 keys, 3670016-byte objects, "
                     "one-shot pressure, and dynamic arm"
                  << std::endl;
        return false;
    }
    return true;
}

std::unique_ptr<datasystem::KVClient> CreateClient(const Config& config, bool exclusive)
{
    datasystem::ConnectOptions options;
    options.host = config.host;
    options.port = config.port;
    options.enableCrossNodeConnection = true;
    options.enableExclusiveConnection = exclusive;
    auto client = std::make_unique<datasystem::KVClient>(options);
    auto status = client->Init();
    if (status.IsError())
    {
        std::cerr << "KVClient Init failed: " << status.ToString() << std::endl;
        return nullptr;
    }
    return client;
}

std::vector<std::string> BuildKeys(const Config& config, uint32_t count)
{
    std::vector<std::string> keys;
    keys.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        keys.emplace_back(config.prefix + "_pressure_" + std::to_string(i));
    }
    return keys;
}

std::vector<std::string> BuildGenerationKeys(const Config& config, uint32_t count, uint32_t generation)
{
    Config generationConfig = config;
    generationConfig.prefix += "_g" + std::to_string(generation);
    return BuildKeys(generationConfig, count);
}

bool PrefillAndVerify(datasystem::KVClient& client, const std::vector<std::string>& keys, const std::string& value)
{
    datasystem::SetParam param;
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    for (const auto& key : keys)
    {
        auto setStatus = client.Set(key, datasystem::StringView(value), param);
        if (setStatus.IsError())
        {
            std::cerr << "prefill failed key=" << key << " detail=" << setStatus.ToString() << std::endl;
            return false;
        }
        datasystem::Optional<datasystem::Buffer> buffer;
        auto getStatus = client.Get(key, buffer, 0);
        if (getStatus.IsError() || !buffer)
        {
            std::cerr << "verification Get failed key=" << key << " detail=" << getStatus.ToString() << std::endl;
            return false;
        }
    }
    return true;
}

bool VerifyKeys(datasystem::KVClient& client, const std::vector<std::string>& keys)
{
    for (const auto& key : keys)
    {
        datasystem::Optional<datasystem::Buffer> buffer;
        auto status = client.Get(key, buffer, 0);
        if (status.IsError() || !buffer)
        {
            std::cerr << "verification-only Get failed key=" << key
                      << " detail=" << status.ToString() << std::endl;
            return false;
        }
    }
    return true;
}

void DeleteKeys(datasystem::KVClient& client, const std::vector<std::string>& keys)
{
    if (keys.empty()) return;
    std::vector<std::string> failed;
    auto status = client.Del(keys, failed);
    if (status.IsError() || !failed.empty())
    {
        std::cerr << "cleanup failed status=" << status.ToString() << " failed_keys=" << failed.size() << std::endl;
    }
}

bool CreateMapping(const Config& config, Mapping* mapping)
{
    mapping->fd = ::open(config.controlPath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0660);
    if (mapping->fd < 0)
    {
        std::perror("open control path");
        return false;
    }
    if (::ftruncate(mapping->fd, sizeof(SharedControl)) != 0)
    {
        std::perror("ftruncate control path");
        return false;
    }
    auto address = ::mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
    if (address == MAP_FAILED)
    {
        std::perror("mmap control path");
        return false;
    }
    mapping->control = static_cast<SharedControl*>(address);
    std::memset(mapping->control, 0, sizeof(SharedControl));
    mapping->control->version = pairec::kvc_burst::kVersion;
    mapping->control->struct_size = sizeof(SharedControl);
    mapping->control->configured_concurrency = config.concurrency;
    mapping->control->pressure_lanes = config.concurrency - 1;
    mapping->control->pressure_key_count
        = config.pressureKeyCount == 0 ? config.concurrency - 1 : config.pressureKeyCount;
    mapping->control->barrier_timeout_ms = config.barrierTimeoutMs;
    mapping->control->pressure_lead_us = config.pressureLeadUs;
    mapping->control->object_size_bytes = config.objectSize;
    pairec::kvc_burst::Store(&mapping->control->trigger_armed, config.initiallyArmed ? 1U : 0U);
    pairec::kvc_burst::Store(&mapping->control->state, static_cast<uint32_t>(State::kStarting));
    pairec::kvc_burst::Store(&mapping->control->magic, pairec::kvc_burst::kMagic);
    return true;
}

bool OpenExistingMapping(const Config& config, Mapping* mapping)
{
    mapping->fd = ::open(config.controlPath.c_str(), O_RDWR);
    if (mapping->fd < 0)
    {
        std::perror("open existing control path");
        return false;
    }
    struct stat info{};
    if (::fstat(mapping->fd, &info) != 0 || info.st_size < static_cast<off_t>(sizeof(SharedControl)))
    {
        std::cerr << "existing control block is too small" << std::endl;
        return false;
    }
    auto address = ::mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
    if (address == MAP_FAILED)
    {
        std::perror("mmap existing control path");
        return false;
    }
    mapping->control = static_cast<SharedControl*>(address);
    return pairec::kvc_burst::IsCompatible(*mapping->control);
}

int ApplyControlAction(const Config& config)
{
    Mapping mapping;
    if (!OpenExistingMapping(config, &mapping)) return 1;
    auto& control = *mapping.control;
    if (config.controlAction == "disarm")
    {
        pairec::kvc_burst::Store(&control.trigger_armed, 0U);
        std::cout << "{\"event\":\"kvc_burst_control\",\"action\":\"disarm\",\"armed\":false}"
                  << std::endl;
        return 0;
    }
    if (config.controlAction == "arm")
    {
        pairec::kvc_burst::Store(&control.trigger_armed, 1U);
        std::cout << "{\"event\":\"kvc_burst_control\",\"action\":\"arm\",\"armed\":true}"
                  << std::endl;
        return 0;
    }
    auto refreshOnly = config.controlAction == "refresh";
    auto refreshAndArm = config.controlAction == "refresh-and-arm";
    auto verifyAndArm = config.controlAction == "verify-and-arm";
    if (!refreshOnly && !refreshAndArm && !verifyAndArm)
    {
        std::cerr << "control_action must be arm, disarm, refresh, refresh-and-arm, or "
                     "verify-and-arm"
                  << std::endl;
        return 2;
    }

    pairec::kvc_burst::Store(&control.trigger_armed, 0U);
    if (pairec::kvc_burst::Load(&control.state) != static_cast<uint32_t>(State::kReady))
    {
        std::cerr << "cannot refresh or verify pressure keys unless burst state is ready"
                  << std::endl;
        return 1;
    }
    auto pressureKeyCount = pairec::kvc_burst::Load(&control.pressure_key_count);
    auto requested = verifyAndArm ? RefreshState::kVerifyRequested : RefreshState::kRequested;
    auto succeeded = verifyAndArm ? RefreshState::kVerifySucceeded : RefreshState::kSucceeded;
    pairec::kvc_burst::Store(&control.refresh_state, static_cast<uint32_t>(requested));
    pairec::kvc_burst::FutexWake(&control.refresh_state);
    auto deadline = pairec::kvc_burst::DeadlineNs(30000);
    uint32_t refreshState = static_cast<uint32_t>(requested);
    while (pairec::kvc_burst::MonotonicNs() < deadline)
    {
        refreshState = pairec::kvc_burst::Load(&control.refresh_state);
        if (refreshState != static_cast<uint32_t>(requested)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (refreshState != static_cast<uint32_t>(succeeded))
    {
        std::cerr << "sidecar pressure key " << (verifyAndArm ? "verification" : "refresh")
                  << " failed or timed out; burst remains disarmed" << std::endl;
        return 1;
    }
    auto arm = refreshAndArm || verifyAndArm;
    pairec::kvc_burst::Store(&control.trigger_armed, arm ? 1U : 0U);
    std::cout << "{\"event\":\"kvc_burst_control\",\"action\":\"" << config.controlAction
              << "\",\"armed\":" << (arm ? "true" : "false") << ",\"keys\":"
              << pressureKeyCount << "}" << std::endl;
    return 0;
}

uint64_t Percentile(std::vector<uint64_t> values, double percentile)
{
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    auto index = static_cast<size_t>((values.size() - 1) * percentile + 0.5);
    return values[std::min(index, values.size() - 1)];
}

// Process-local per-lane sustained pressure statistics. These do not need to
// live in SharedControl because the workers and the aggregator are threads of
// the same process; each lane only writes its own slot and the aggregator
// reads them after completed_pressure_lanes reaches the lane count.
struct SustainedLaneStats
{
    uint32_t loops{0};
    uint32_t errors{0};
    uint64_t lastEndNs{0};
    uint32_t stopReason{0};
};

struct SustainedReport
{
    bool enabled{false};
    uint32_t maxDurationMs{0};
    uint32_t maxLoops{0};
    uint64_t loopGets{0};
    uint64_t errors{0};
    uint32_t minLaneLoops{0};
    uint32_t maxLaneLoops{0};
    uint32_t stopBusiness{0};
    uint32_t stopMaxDuration{0};
    uint32_t stopMaxLoops{0};
    uint64_t windowUs{0};
};

struct Aggregate
{
    bool valid{false};
    Failure failure{Failure::kNone};
    uint32_t maxActiveAll{0};
    uint32_t businessOverlap{0};
    uint64_t startSkewUs{0};
    uint64_t businessGetUs{0};
    uint32_t businessSubmitRank{0};
    uint32_t pressureStartedBeforeBusiness{0};
    uint32_t pressureInflightAtBusinessStart{0};
    uint32_t pressureCompletedBeforeBusiness{0};
    uint64_t pressureAvgUs{0};
    uint64_t pressureP95Us{0};
    uint64_t pressureP99Us{0};
    uint64_t pressureMaxUs{0};
};

Aggregate AggregateResult(SharedControl& control)
{
    Aggregate result;
    auto pressureLanes = pairec::kvc_burst::Load(&control.pressure_lanes);
    auto businessStart = pairec::kvc_burst::Load(&control.business_start_ns);
    auto businessEnd = pairec::kvc_burst::Load(&control.business_end_ns);
    result.businessGetUs = businessEnd > businessStart ? (businessEnd - businessStart) / 1000ULL : 0;
    result.businessSubmitRank = pairec::kvc_burst::BusinessSubmitRank(control);

    std::vector<uint64_t> starts;
    std::vector<uint64_t> pressureLatencies;
    std::vector<std::pair<uint64_t, int>> events;
    if (businessStart > 0 && businessEnd >= businessStart)
    {
        starts.push_back(businessStart);
        events.emplace_back(businessStart, 1);
        events.emplace_back(businessEnd, -1);
    }
    uint64_t latencySum = 0;
    for (uint32_t i = 0; i < pressureLanes; ++i)
    {
        auto start = pairec::kvc_burst::Load(&control.pressure_start_ns[i]);
        auto end = pairec::kvc_burst::Load(&control.pressure_end_ns[i]);
        if (start == 0 || end < start) continue;
        starts.push_back(start);
        events.emplace_back(start, 1);
        events.emplace_back(end, -1);
        auto latency = (end - start) / 1000ULL;
        pressureLatencies.push_back(latency);
        latencySum += latency;
        if (businessStart > 0 && start < businessStart)
        {
            ++result.pressureStartedBeforeBusiness;
            if (end > businessStart)
            {
                ++result.pressureInflightAtBusinessStart;
            }
            else
            {
                ++result.pressureCompletedBeforeBusiness;
            }
        }
        if (businessStart < end && start < businessEnd)
        {
            ++result.businessOverlap;
        }
    }
    if (!starts.empty())
    {
        auto [minimum, maximum] = std::minmax_element(starts.begin(), starts.end());
        result.startSkewUs = (*maximum - *minimum) / 1000ULL;
    }
    std::sort(events.begin(), events.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first < right.first;
        return left.second > right.second;
    });
    uint32_t active = 0;
    for (const auto& event : events)
    {
        if (event.second > 0)
        {
            active += 1;
            result.maxActiveAll = std::max(result.maxActiveAll, active);
        }
        else if (active > 0)
        {
            active -= 1;
        }
    }
    if (!pressureLatencies.empty())
    {
        result.pressureAvgUs = latencySum / pressureLatencies.size();
        result.pressureP95Us = Percentile(pressureLatencies, 0.95);
        result.pressureP99Us = Percentile(pressureLatencies, 0.99);
        result.pressureMaxUs = *std::max_element(pressureLatencies.begin(), pressureLatencies.end());
    }

    auto concurrency = pairec::kvc_burst::Load(&control.configured_concurrency);
    auto requiredActive = (concurrency * 95U + 99U) / 100U;
    auto requiredOverlap = (pressureLanes * 95U + 99U) / 100U;
    if (pairec::kvc_burst::Load(&control.barrier_failed) != 0)
    {
        result.failure = Failure::kBarrierTimeout;
    }
    else if (pairec::kvc_burst::Load(&control.business_success) == 0)
    {
        result.failure = Failure::kBusinessGetFailed;
    }
    else if (pairec::kvc_burst::Load(&control.pressure_errors) != 0
        || pairec::kvc_burst::Load(&control.pressure_success) != pressureLanes)
    {
        result.failure = Failure::kPressureGetFailed;
    }
    else if (pairec::kvc_burst::Load(&control.pressure_first_failed) != 0
        || !pairec::kvc_burst::PressureFirstSatisfied(pressureLanes,
            result.pressureStartedBeforeBusiness, result.pressureInflightAtBusinessStart,
            result.businessSubmitRank))
    {
        result.failure = Failure::kPressureNotEstablished;
    }
    else if (result.maxActiveAll < requiredActive || result.businessOverlap < requiredOverlap)
    {
        result.failure = Failure::kInvalidControl;
    }
    else
    {
        result.valid = true;
    }
    return result;
}

SustainedReport SummarizeSustained(const Config& config, const SharedControl& control,
    const std::vector<SustainedLaneStats>& lanes)
{
    SustainedReport report;
    report.enabled = config.sustainedPressure;
    report.maxDurationMs = config.sustainedMaxDurationMs;
    report.maxLoops = config.sustainedMaxLoops;
    if (!config.sustainedPressure)
    {
        return report;
    }
    auto pressureLanes = pairec::kvc_burst::Load(&control.pressure_lanes);
    bool haveLane = false;
    uint64_t minStart = 0;
    uint64_t maxEnd = 0;
    for (uint32_t i = 0; i < pressureLanes && i < lanes.size(); ++i)
    {
        const auto& lane = lanes[i];
        report.loopGets += lane.loops;
        report.errors += lane.errors;
        if (!haveLane)
        {
            report.minLaneLoops = lane.loops;
            report.maxLaneLoops = lane.loops;
            haveLane = true;
        }
        else
        {
            report.minLaneLoops = std::min(report.minLaneLoops, lane.loops);
            report.maxLaneLoops = std::max(report.maxLaneLoops, lane.loops);
        }
        switch (static_cast<pairec::kvc_burst::SustainedStop>(lane.stopReason))
        {
        case pairec::kvc_burst::SustainedStop::kBusinessDone: ++report.stopBusiness; break;
        case pairec::kvc_burst::SustainedStop::kMaxDuration: ++report.stopMaxDuration; break;
        case pairec::kvc_burst::SustainedStop::kMaxLoops: ++report.stopMaxLoops; break;
        default: break;
        }
        auto laneStart = pairec::kvc_burst::Load(&control.pressure_start_ns[i]);
        auto laneEnd = lane.lastEndNs != 0
            ? lane.lastEndNs : pairec::kvc_burst::Load(&control.pressure_end_ns[i]);
        if (laneStart > 0 && laneEnd >= laneStart)
        {
            minStart = minStart == 0 ? laneStart : std::min(minStart, laneStart);
            maxEnd = std::max(maxEnd, laneEnd);
        }
    }
    if (minStart > 0 && maxEnd > minStart)
    {
        report.windowUs = (maxEnd - minStart) / 1000ULL;
    }
    return report;
}

std::string JsonResult(const Config& config, const SharedControl& control, uint32_t generation,
    const Aggregate& result, const SustainedReport& sustained)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "{\"event\":\"kvc_burst_result\",\"generation\":" << generation
           << ",\"request_id\":\"" << control.request_id << "\",\"concurrency\":"
           << control.configured_concurrency << ",\"pressure_lanes\":" << control.pressure_lanes
           << ",\"pressure_key_count\":" << control.pressure_key_count
           << ",\"pressure_engine\":\""
           << (config.inProcessPressure ? "inprocess-shared-client" : "sidecar-exclusive-clients")
           << "\",\"shared_client_errors\":"
           << (config.inProcessPressure ? control.pressure_errors : 0U)
           << ",\"object_size_bytes\":" << config.objectSize << ",\"shuffle_seed\":"
           << control.shuffle_seed << ",\"barrier_wait_ms\":"
           << static_cast<double>(control.business_barrier_wait_us) / 1000.0
           << ",\"pressure_first_wait_ms\":"
           << static_cast<double>(control.business_pressure_wait_us) / 1000.0
           << ",\"pressure_lead_wait_ms\":"
           << static_cast<double>(control.business_lead_wait_us) / 1000.0
           << ",\"coordination_wait_ms\":"
           << static_cast<double>(control.business_barrier_wait_us
                  + control.business_pressure_wait_us + control.business_lead_wait_us)
                  / 1000.0
           << ",\"trigger_us\":" << control.business_trigger_us
           << ",\"business_api\":\""
           << pairec::kvc_burst::BusinessApiName(
                  static_cast<pairec::kvc_burst::BusinessApi>(control.business_api))
           << "\",\"business_key_count\":" << control.business_key_count
           << ",\"business_bytes\":" << control.business_bytes
           << ",\"business_get_ms\":" << static_cast<double>(result.businessGetUs) / 1000.0
           << ",\"pressure_get_avg_ms\":" << static_cast<double>(result.pressureAvgUs) / 1000.0
           << ",\"pressure_get_p95_ms\":" << static_cast<double>(result.pressureP95Us) / 1000.0
           << ",\"pressure_get_p99_ms\":" << static_cast<double>(result.pressureP99Us) / 1000.0
           << ",\"pressure_get_max_ms\":" << static_cast<double>(result.pressureMaxUs) / 1000.0
           << ",\"start_skew_us\":" << result.startSkewUs << ",\"max_active_all_gets\":"
           << result.maxActiveAll << ",\"business_overlap_gets\":" << result.businessOverlap
           << ",\"business_submit_rank\":" << result.businessSubmitRank
           << ",\"pressure_started_before_business\":"
           << result.pressureStartedBeforeBusiness
           << ",\"pressure_inflight_at_business_start\":"
           << result.pressureInflightAtBusinessStart
           << ",\"pressure_completed_before_business\":"
           << result.pressureCompletedBeforeBusiness
           << ",\"pressure_first_required\":"
           << pairec::kvc_burst::RequiredPressureFirst(control.pressure_lanes);
    {
        auto businessStart = pairec::kvc_burst::Load(&control.business_start_ns);
        auto pressureLanes = pairec::kvc_burst::Load(&control.pressure_lanes);
        output << ",\"pressure_start_offsets_us\":";
        bool firstOffset = true;
        output << '[';
        for (uint32_t i = 0; i < pressureLanes; ++i)
        {
            if (!firstOffset) output << ',';
            firstOffset = false;
            auto start = pairec::kvc_burst::Load(&control.pressure_start_ns[i]);
            if (start == 0 || businessStart == 0) output << "null";
            else if (start >= businessStart) output << static_cast<long long>(start - businessStart) / 1000LL;
            else output << -static_cast<long long>(businessStart - start) / 1000LL;
        }
        output << "],\"pressure_end_offsets_us\":";
        firstOffset = true;
        output << '[';
        for (uint32_t i = 0; i < pressureLanes; ++i)
        {
            if (!firstOffset) output << ',';
            firstOffset = false;
            auto end = pairec::kvc_burst::Load(&control.pressure_end_ns[i]);
            if (end == 0 || businessStart == 0) output << "null";
            else if (end >= businessStart) output << static_cast<long long>(end - businessStart) / 1000LL;
            else output << -static_cast<long long>(businessStart - end) / 1000LL;
        }
        output << ']';
    }
    output << ",\"sustained_enabled\":" << (sustained.enabled ? "true" : "false")
           << ",\"sustained_max_duration_ms\":" << sustained.maxDurationMs
           << ",\"sustained_max_loops_config\":" << sustained.maxLoops
           << ",\"sustained_loop_gets\":" << sustained.loopGets
           << ",\"sustained_errors\":" << sustained.errors
           << ",\"sustained_min_lane_loops\":" << sustained.minLaneLoops
           << ",\"sustained_max_lane_loops\":" << sustained.maxLaneLoops
           << ",\"sustained_stop_business\":" << sustained.stopBusiness
           << ",\"sustained_stop_max_duration\":" << sustained.stopMaxDuration
           << ",\"sustained_stop_max_loops\":" << sustained.stopMaxLoops
           << ",\"sustained_window_ms\":" << static_cast<double>(sustained.windowUs) / 1000.0
           << ",\"pressure_success\":" << control.pressure_success << ",\"pressure_errors\":"
           << control.pressure_errors << ",\"business_success\":"
           << (control.business_success ? "true" : "false") << ",\"valid\":"
           << (result.valid ? "true" : "false") << ",\"failure\":"
           << static_cast<uint32_t>(result.failure) << "}";
    return output.str();
}

void WriteLine(const Config& config, const std::string& line)
{
    std::cout << line << std::endl;
    if (!config.statsFile.empty())
    {
        std::ofstream output(config.statsFile, std::ios::app);
        output << line << '\n';
    }
}

bool PublishReadyFile(const std::string& path, const std::string& contents)
{
    if (path.empty()) return true;
    auto temporary = path + ".tmp." + std::to_string(::getpid());
    auto fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
    {
        std::perror("open ready file");
        return false;
    }
    size_t written = 0;
    while (written < contents.size())
    {
        auto count = ::write(fd, contents.data() + written, contents.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0)
        {
            std::perror("write ready file");
            ::close(fd);
            std::remove(temporary.c_str());
            return false;
        }
        written += static_cast<size_t>(count);
    }
    if (::fsync(fd) != 0)
    {
        std::perror("fsync ready file");
        ::close(fd);
        std::remove(temporary.c_str());
        return false;
    }
    if (::close(fd) != 0)
    {
        std::perror("close ready file");
        std::remove(temporary.c_str());
        return false;
    }
    if (::rename(temporary.c_str(), path.c_str()) != 0)
    {
        std::perror("rename ready file");
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

void ResetGeneration(SharedControl& control, uint32_t generation, uint64_t shuffleSeed)
{
    pairec::kvc_burst::Store(&control.arrived_participants, 0U);
    pairec::kvc_burst::Store(&control.pressure_started_lanes, 0U);
    pairec::kvc_burst::Store(&control.pressure_first_failed, 0U);
    pairec::kvc_burst::Store(&control.business_release_generation, 0U);
    pairec::kvc_burst::Store(&control.completed_pressure_lanes, 0U);
    pairec::kvc_burst::Store(&control.barrier_failed, 0U);
    pairec::kvc_burst::Store(&control.business_done_generation, 0U);
    pairec::kvc_burst::Store(&control.business_success, 0U);
    pairec::kvc_burst::Store(&control.pressure_success, 0U);
    pairec::kvc_burst::Store(&control.pressure_errors, 0U);
    pairec::kvc_burst::Store(&control.business_barrier_wait_us, uint64_t{0});
    pairec::kvc_burst::Store(&control.business_pressure_wait_us, uint64_t{0});
    pairec::kvc_burst::Store(&control.business_lead_wait_us, uint64_t{0});
    pairec::kvc_burst::Store(&control.business_trigger_us, uint64_t{0});
    pairec::kvc_burst::Store(&control.business_bytes, uint64_t{0});
    pairec::kvc_burst::Store(&control.trigger_started_ns, uint64_t{0});
    pairec::kvc_burst::Store(&control.trigger_completed_ns, uint64_t{0});
    pairec::kvc_burst::Store(
        &control.business_api, static_cast<uint32_t>(pairec::kvc_burst::BusinessApi::kUnknown));
    pairec::kvc_burst::Store(&control.business_key_count, 0U);
    pairec::kvc_burst::Store(
        &control.business_trigger_status, static_cast<uint32_t>(pairec::kvc_burst::TriggerStatus::kNone));
    pairec::kvc_burst::Store(&control.claim_started_ns, uint64_t{0});
    pairec::kvc_burst::Store(&control.business_start_ns, uint64_t{0});
    pairec::kvc_burst::Store(&control.business_end_ns, uint64_t{0});
    pairec::kvc_burst::Store(&control.shuffle_seed, shuffleSeed);
    pairec::kvc_burst::CopyRequestId(control.request_id, nullptr);
    for (uint32_t i = 0; i < pairec::kvc_burst::kMaxPressureLanes; ++i)
    {
        pairec::kvc_burst::Store(&control.pressure_start_ns[i], uint64_t{0});
        pairec::kvc_burst::Store(&control.pressure_end_ns[i], uint64_t{0});
        pairec::kvc_burst::Store(&control.pressure_ok[i], 0U);
    }
    pairec::kvc_burst::Store(&control.prepared_generation, generation);
}

void PressureWorker(uint32_t lane, SharedControl* control, datasystem::KVClient* client,
    const std::vector<std::string>* keys, const std::vector<uint32_t>* permutation,
    const Config* config, SustainedLaneStats* sustainedStats)
{
    uint32_t previousGeneration = 0;
    while (!gStop.load(std::memory_order_relaxed))
    {
        pairec::kvc_burst::FetchAdd(&control->armed_pressure_lanes, 1U);
        pairec::kvc_burst::FutexWake(&control->armed_pressure_lanes);
        uint32_t generation = 0;
        if (!pairec::kvc_burst::WaitForGenerationChange(&control->run_generation, previousGeneration, &generation))
        {
            break;
        }
        pairec::kvc_burst::FetchSub(&control->armed_pressure_lanes, 1U);
        if (gStop.load(std::memory_order_relaxed)
            || pairec::kvc_burst::Load(&control->state) == static_cast<uint32_t>(State::kStopping))
        {
            break;
        }
        previousGeneration = generation;
        uint64_t barrierWaitUs = 0;
        auto released = pairec::kvc_burst::ArriveAndWait(
            control, generation, pairec::kvc_burst::Load(&control->barrier_timeout_ms), &barrierWaitUs);
        bool ok = false;
        if (released)
        {
            auto start = pairec::kvc_burst::MonotonicNs();
            pairec::kvc_burst::Store(&control->pressure_start_ns[lane], start);
            pairec::kvc_burst::FetchAdd(&control->pressure_started_lanes, 1U);
            pairec::kvc_burst::FutexWake(&control->pressure_started_lanes);
            datasystem::Optional<datasystem::Buffer> buffer;
            auto status = client->Get((*keys)[(*permutation)[lane]], buffer, 0);
            ok = !status.IsError() && static_cast<bool>(buffer);
            if (!ok)
            {
                std::lock_guard<std::mutex> lock(gPressureErrorMutex);
                std::cerr << "pressure Get failed generation=" << generation << " lane=" << lane
                          << " key=" << (*keys)[(*permutation)[lane]]
                          << " has_buffer=" << (buffer ? "true" : "false")
                          << " detail=" << status.ToString() << std::endl;
            }
            pairec::kvc_burst::Store(&control->pressure_end_ns[lane], pairec::kvc_burst::MonotonicNs());
        }
        if (released && config->sustainedPressure)
        {
            // Keep issuing Gets on this lane's key until the business Get for
            // this generation completes. The loop and duration budgets bound
            // the burst when the business request fails or crashes.
            auto loopStarted = pairec::kvc_burst::MonotonicNs();
            auto stop = pairec::kvc_burst::SustainedStop::kNone;
            bool errorLogged = false;
            while (!gStop.load(std::memory_order_relaxed))
            {
                stop = pairec::kvc_burst::SustainedStopReason(*control, generation, loopStarted,
                    sustainedStats->loops, config->sustainedMaxDurationMs, config->sustainedMaxLoops);
                if (stop != pairec::kvc_burst::SustainedStop::kNone)
                {
                    break;
                }
                datasystem::Optional<datasystem::Buffer> buffer;
                auto status = client->Get((*keys)[(*permutation)[lane]], buffer, 0);
                bool sustainedOk = !status.IsError() && static_cast<bool>(buffer);
                sustainedStats->loops += 1;
                if (!sustainedOk)
                {
                    sustainedStats->errors += 1;
                    if (!errorLogged)
                    {
                        errorLogged = true;
                        std::lock_guard<std::mutex> lock(gPressureErrorMutex);
                        std::cerr << "sustained pressure Get failed generation=" << generation
                                  << " lane=" << lane << " key=" << (*keys)[(*permutation)[lane]]
                                  << " detail=" << status.ToString() << std::endl;
                    }
                }
                sustainedStats->lastEndNs = pairec::kvc_burst::MonotonicNs();
            }
            sustainedStats->stopReason = static_cast<uint32_t>(stop);
        }
        pairec::kvc_burst::Store(&control->pressure_ok[lane], ok ? 1U : 0U);
        pairec::kvc_burst::FetchAdd(ok ? &control->pressure_success : &control->pressure_errors, 1U);
        pairec::kvc_burst::FetchAdd(&control->completed_pressure_lanes, 1U);
        pairec::kvc_burst::FutexWake(&control->completed_pressure_lanes);
    }
}

} // namespace

int Run(int argc, char** argv)
{
    gStartupStage = "parse_args";
    Config config;
    if (!ParseArgs(argc, argv, &config)) return 2;
    if (!config.controlAction.empty()) return ApplyControlAction(config);
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    gStartupStage = "create_mapping";
    Mapping mapping;
    if (!CreateMapping(config, &mapping)) return 1;
    auto& control = *mapping.control;
    auto pressureLanes = config.concurrency - 1;
    auto pressureKeyCount = config.pressureKeyCount == 0 ? pressureLanes : config.pressureKeyCount;
    uint32_t pressureKeyGeneration = 0;
    auto keys = BuildGenerationKeys(config, pressureKeyCount, pressureKeyGeneration);
    std::string value(config.objectSize, 'k');

    std::unique_ptr<datasystem::KVClient> controlClient;
    if (pressureLanes > 0)
    {
        gStartupStage = "create_control_client";
        controlClient = CreateClient(config, false);
        if (!controlClient) return 1;
        gStartupStage = "prefill_and_verify";
        if (!PrefillAndVerify(*controlClient, keys, value)) return 1;
    }
    pairec::kvc_burst::Store(&control.keys_verified, pressureLanes);

    std::vector<std::unique_ptr<datasystem::KVClient>> clients;
    clients.reserve(pressureLanes);
    for (uint32_t i = 0; !config.inProcessPressure && i < pressureLanes; ++i)
    {
        gStartupStage = "create_pressure_clients";
        auto client = CreateClient(config, true);
        if (!client) return 1;
        clients.emplace_back(std::move(client));
    }
    pairec::kvc_burst::Store(
        &control.clients_connected, config.inProcessPressure ? 0U : pressureLanes);

    std::vector<uint32_t> permutation(pressureLanes);
    for (uint32_t i = 0; !config.inProcessPressure && i < pressureLanes; ++i)
    {
        permutation[i] = i % pressureKeyCount;
    }
    std::vector<SustainedLaneStats> sustainedStats(pressureLanes);
    std::vector<std::thread> workers;
    workers.reserve(pressureLanes);
    for (uint32_t i = 0; !config.inProcessPressure && i < pressureLanes; ++i)
    {
        gStartupStage = "start_pressure_workers";
        workers.emplace_back(PressureWorker, i, &control, clients[i].get(), &keys, &permutation,
            &config, &sustainedStats[i]);
    }

    while (!config.inProcessPressure && !gStop.load(std::memory_order_relaxed)
        && pairec::kvc_burst::Load(&control.armed_pressure_lanes) != pressureLanes)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (gStop.load(std::memory_order_relaxed)) return 1;

    uint32_t generation = 1;
    auto prepare = [&]() {
        auto shuffleSeed = config.seed + generation;
        std::mt19937_64 random(shuffleSeed);
        std::shuffle(permutation.begin(), permutation.end(), random);
        ResetGeneration(control, generation, shuffleSeed);
        pairec::kvc_burst::Store(&control.state, static_cast<uint32_t>(State::kReady));
        pairec::kvc_burst::FutexWake(&control.state);
    };
    prepare();
    gStartupStage = "ready";

    if (!config.readyFile.empty())
    {
        auto ready = std::string{"pid="} + std::to_string(::getpid()) + " concurrency="
            + std::to_string(config.concurrency) + " pressure_lanes=" + std::to_string(pressureLanes)
            + " pressure_key_count=" + std::to_string(pressureKeyCount)
            + " keys_verified=" + std::to_string(pressureLanes) + " clients_connected="
            + std::to_string(config.inProcessPressure ? 0U : pressureLanes)
            + " armed_workers=" + std::to_string(config.inProcessPressure ? 0U : pressureLanes)
            + " pressure_engine="
            + (config.inProcessPressure ? "inprocess-shared-client" : "sidecar-exclusive-clients")
            + "\n";
        if (!PublishReadyFile(config.readyFile, ready))
        {
            pairec::kvc_burst::Store(&control.state, static_cast<uint32_t>(State::kFailed));
            return 1;
        }
    }
    WriteLine(config, "{\"event\":\"kvc_burst_ready\",\"version\":4,\"trigger_operation\":\"get\",\"concurrency\":" + std::to_string(config.concurrency)
            + ",\"pressure_lanes\":" + std::to_string(pressureLanes) + ",\"object_size_bytes\":"
            + std::to_string(config.objectSize) + ",\"pressure_key_count\":" + std::to_string(pressureKeyCount)
            + ",\"clients_connected\":" + std::to_string(config.inProcessPressure ? 0U : pressureLanes)
            + ",\"keys_verified\":" + std::to_string(pressureLanes)
            + ",\"pressure_engine\":\""
            + (config.inProcessPressure ? "inprocess-shared-client" : "sidecar-exclusive-clients") + "\""
            + ",\"pressure_lead_us\":" + std::to_string(config.pressureLeadUs)
            + ",\"sustained_pressure\":" + (config.sustainedPressure ? "true" : "false")
            + ",\"sustained_max_duration_ms\":" + std::to_string(config.sustainedMaxDurationMs)
            + ",\"sustained_max_loops\":" + std::to_string(config.sustainedMaxLoops) + "}");

    while (!gStop.load(std::memory_order_relaxed))
    {
        pairec::kvc_burst::Store(&control.heartbeat_ns, pairec::kvc_burst::MonotonicNs());
        auto state = pairec::kvc_burst::Load(&control.state);
        auto refreshState = pairec::kvc_burst::Load(&control.refresh_state);
        auto refreshRequested = refreshState == static_cast<uint32_t>(RefreshState::kRequested);
        auto verifyRequested
            = refreshState == static_cast<uint32_t>(RefreshState::kVerifyRequested);
        if (state == static_cast<uint32_t>(State::kReady)
            && (refreshRequested || verifyRequested))
        {
            if (verifyRequested)
            {
                auto verified = pressureKeyCount == 0 || VerifyKeys(*controlClient, keys);
                pairec::kvc_burst::Store(&control.refresh_state,
                    static_cast<uint32_t>(verified ? RefreshState::kVerifySucceeded
                                                   : RefreshState::kVerifyFailed));
                pairec::kvc_burst::FutexWake(&control.refresh_state);
                if (verified)
                {
                    std::cout << "{\"event\":\"kvc_burst_keys_verified\",\"generation\":"
                              << pressureKeyGeneration << ",\"pressure_key_count\":"
                              << pressureKeyCount << "}" << std::endl;
                }
                continue;
            }

            ++pressureKeyGeneration;
            auto refreshed = pressureKeyCount == 0;
            std::vector<std::string> refreshedKeys;
            if (pressureKeyCount > 0)
            {
                DeleteKeys(*controlClient, keys);
                refreshedKeys = BuildGenerationKeys(config, pressureKeyCount, pressureKeyGeneration);
                refreshed = PrefillAndVerify(*controlClient, refreshedKeys, value);
                if (!refreshed) DeleteKeys(*controlClient, refreshedKeys);
            }
            if (refreshed)
            {
                keys = std::move(refreshedKeys);
                std::cout << "{\"event\":\"kvc_burst_keys_refreshed\",\"generation\":"
                          << pressureKeyGeneration << ",\"pressure_key_count\":"
                          << pressureKeyCount << "}" << std::endl;
            }
            pairec::kvc_burst::Store(&control.refresh_state,
                static_cast<uint32_t>(refreshed ? RefreshState::kSucceeded : RefreshState::kFailed));
            pairec::kvc_burst::FutexWake(&control.refresh_state);
            continue;
        }
        if (state == static_cast<uint32_t>(State::kClaimed))
        {
            auto claimedAt = pairec::kvc_burst::Load(&control.claim_started_ns);
            auto claimTimeoutNs = static_cast<uint64_t>(config.barrierTimeoutMs) * 1000000ULL;
            if (claimedAt > 0 && pairec::kvc_burst::MonotonicNs() - claimedAt >= claimTimeoutNs)
            {
                pairec::kvc_burst::Store(&control.result_failure,
                    static_cast<uint32_t>(Failure::kBarrierTimeout));
                pairec::kvc_burst::Store(&control.state, static_cast<uint32_t>(State::kFailed));
                if (!config.readyFile.empty()) std::remove(config.readyFile.c_str());
                break;
            }
        }
        if (state != static_cast<uint32_t>(State::kRunning))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto runningGeneration = pairec::kvc_burst::Load(&control.run_generation);
        auto deadline = pairec::kvc_burst::DeadlineNs(120000);
        while (!gStop.load(std::memory_order_relaxed)
            && (pairec::kvc_burst::Load(&control.completed_pressure_lanes) != pressureLanes
                || pairec::kvc_burst::Load(&control.business_done_generation) != runningGeneration)
            && pairec::kvc_burst::MonotonicNs() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto result = AggregateResult(control);
        auto sustainedReport = SummarizeSustained(config, control, sustainedStats);
        pairec::kvc_burst::Store(&control.result_valid, result.valid ? 1U : 0U);
        pairec::kvc_burst::Store(&control.result_failure, static_cast<uint32_t>(result.failure));
        pairec::kvc_burst::Store(&control.result_max_active_all, result.maxActiveAll);
        pairec::kvc_burst::Store(&control.result_business_overlap, result.businessOverlap);
        pairec::kvc_burst::Store(&control.result_business_submit_rank, result.businessSubmitRank);
        pairec::kvc_burst::Store(&control.result_pressure_started_before_business,
            result.pressureStartedBeforeBusiness);
        pairec::kvc_burst::Store(&control.result_pressure_inflight_at_business_start,
            result.pressureInflightAtBusinessStart);
        pairec::kvc_burst::Store(&control.result_pressure_completed_before_business,
            result.pressureCompletedBeforeBusiness);
        pairec::kvc_burst::Store(&control.result_start_skew_us, result.startSkewUs);
        pairec::kvc_burst::Store(&control.result_business_get_us, result.businessGetUs);
        pairec::kvc_burst::Store(&control.result_pressure_avg_us, result.pressureAvgUs);
        pairec::kvc_burst::Store(&control.result_pressure_p95_us, result.pressureP95Us);
        pairec::kvc_burst::Store(&control.result_pressure_p99_us, result.pressureP99Us);
        pairec::kvc_burst::Store(&control.result_pressure_max_us, result.pressureMaxUs);
        auto resultJson = JsonResult(config, control, runningGeneration, result, sustainedReport);
        auto marker = std::string{"\"event\":\"kvc_burst_result\""};
        resultJson.replace(resultJson.find(marker), marker.size(), "\"event\":\"kvc_burst_complete\"");
        WriteLine(config, resultJson);
        pairec::kvc_burst::Store(&control.result_generation, runningGeneration);
        pairec::kvc_burst::FutexWake(&control.result_generation);

        if (result.failure == Failure::kBarrierTimeout || result.failure == Failure::kPressureGetFailed)
        {
            pairec::kvc_burst::Store(&control.state, static_cast<uint32_t>(State::kFailed));
            if (!config.readyFile.empty()) std::remove(config.readyFile.c_str());
            break;
        }
        while (!config.inProcessPressure && !gStop.load(std::memory_order_relaxed)
            && pairec::kvc_burst::Load(&control.armed_pressure_lanes) != pressureLanes)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ++generation;
        prepare();
    }

    pairec::kvc_burst::Store(&control.state, static_cast<uint32_t>(State::kStopping));
    gStop.store(true, std::memory_order_relaxed);
    pairec::kvc_burst::FetchAdd(&control.run_generation, 1U);
    pairec::kvc_burst::FutexWake(&control.run_generation);
    pairec::kvc_burst::FutexWake(&control.release_generation);
    for (auto& worker : workers)
    {
        if (worker.joinable()) worker.join();
    }
    if (!config.readyFile.empty()) std::remove(config.readyFile.c_str());
    if (config.cleanupKeys && controlClient) DeleteKeys(*controlClient, keys);
    return pairec::kvc_burst::Load(&control.barrier_failed) == 0 ? 0 : 1;
}

int main(int argc, char** argv)
{
    try
    {
        return Run(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "{\"event\":\"kvc_burst_startup_failed\",\"stage\":\"" << gStartupStage
                  << "\",\"error\":\"" << error.what() << "\"}" << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "{\"event\":\"kvc_burst_startup_failed\",\"stage\":\"" << gStartupStage
                  << "\",\"error\":\"unknown exception\"}" << std::endl;
        return 1;
    }
}
