#include "kvc_operation_proxy.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

extern "C" char const* pairecKvcBurstProxyProtocolMarker()
{
    return "PAIREC_KVC_BURST_PROXY_V5";
}

extern "C" char const* pairecKvcInProcessBurstCapabilityMarker()
{
    return "PAIREC_KVC_INPROCESS_SUSTAINED_C32_V1";
}

namespace pairec::kvc_burst
{
namespace
{

bool parseEnabled(char const* value)
{
    if (value == nullptr) return false;
    std::string const text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

bool measureDisabled()
{
    static bool const enabled = parseEnabled(std::getenv("KVC_BURST_MEASURE_DISABLED"));
    return enabled;
}

bool verboseEvents()
{
    static bool const enabled = parseEnabled(std::getenv("KVC_BURST_VERBOSE"));
    return enabled;
}

bool inProcessPressureEnabled()
{
    return parseEnabled(std::getenv("PAIREC_KVC_INPROCESS_BURST"));
}

std::string pressurePrefix()
{
    auto const* configured = std::getenv("PAIREC_KVC_INPROCESS_PRESSURE_PREFIX");
    return configured != nullptr ? configured : "PairecKvcBurstV2";
}

std::string jsonEscape(std::string const& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value)
    {
        if (ch == '\\' || ch == '"') escaped.push_back('\\');
        if (ch == '\n') escaped += "\\n";
        else if (ch == '\r') escaped += "\\r";
        else escaped.push_back(ch);
    }
    return escaped;
}

class ControlMapping
{
public:
    ~ControlMapping()
    {
        if (mControl != nullptr) ::munmap(mControl, sizeof(SharedControl));
        if (mFd >= 0) ::close(mFd);
    }

    SharedControl* get()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mControl != nullptr && IsCompatible(*mControl)) return mControl;
        closeUnlocked();
        auto const* configured = std::getenv("KVC_BURST_CONTROL_PATH");
        std::string const path = configured != nullptr ? configured : "/run/pairec-kvc-burst/control";
        mFd = ::open(path.c_str(), O_RDWR);
        if (mFd < 0) return nullptr;
        struct stat info{};
        if (::fstat(mFd, &info) != 0 || info.st_size < static_cast<off_t>(sizeof(SharedControl)))
        {
            closeUnlocked();
            return nullptr;
        }
        auto address = ::mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, mFd, 0);
        if (address == MAP_FAILED)
        {
            mControl = nullptr;
            closeUnlocked();
            return nullptr;
        }
        mControl = static_cast<SharedControl*>(address);
        if (!IsCompatible(*mControl))
        {
            closeUnlocked();
            return nullptr;
        }
        return mControl;
    }

private:
    void closeUnlocked()
    {
        if (mControl != nullptr)
        {
            ::munmap(mControl, sizeof(SharedControl));
            mControl = nullptr;
        }
        if (mFd >= 0)
        {
            ::close(mFd);
            mFd = -1;
        }
    }

    std::mutex mMutex;
    int mFd{-1};
    SharedControl* mControl{nullptr};
};

ControlMapping& mapping()
{
    static ControlMapping value;
    return value;
}

std::mutex& claimMutex()
{
    static std::mutex value;
    return value;
}

class SeenRequests
{
public:
    bool contains(std::string const& requestId) const { return mIds.count(requestId) != 0; }

    void insert(std::string const& requestId)
    {
        if (!mIds.insert(requestId).second) return;
        mOrder.push_back(requestId);
        if (mOrder.size() > kCapacity)
        {
            mIds.erase(mOrder.front());
            mOrder.pop_front();
        }
    }

private:
    static constexpr size_t kCapacity = 4096;
    std::deque<std::string> mOrder;
    std::unordered_set<std::string> mIds;
};

SeenRequests& seenRequests()
{
    static SeenRequests value;
    return value;
}

void emitSkipped(std::string const& requestId, TriggerStatus status)
{
    if (!verboseEvents()) return;
    std::cout << "{\"event\":\"kvc_proxy_get_skipped\",\"request_id\":\""
              << jsonEscape(requestId) << "\",\"status\":" << static_cast<uint32_t>(status) << "}"
              << std::endl;
}

} // namespace

bool kvcBurstEnabled()
{
    static bool const enabled = parseEnabled(std::getenv("KVC_BURST_ENABLED"));
    return enabled;
}

class InProcessPressureCoordinator
{
public:
    ~InProcessPressureCoordinator()
    {
        mStopping.store(true, std::memory_order_release);
        if (auto* control = mapping().get())
        {
            FetchAdd(&control->pressure_command_generation, 1U);
            FutexWake(&control->pressure_command_generation);
        }
        for (auto& worker : mWorkers)
        {
            if (worker.joinable()) worker.join();
        }
    }

    bool registerClient(InProcessPressureGet get)
    {
        if (!inProcessPressureEnabled() || !get) return false;
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mGet)
        {
            auto* control = mapping().get();
            if (control == nullptr) return false;
            mGet = std::make_shared<InProcessPressureGet>(std::move(get));
            auto const lanes = Load(&control->pressure_lanes);
            mWorkers.reserve(lanes);
            try
            {
                for (uint32_t lane = 0; lane < lanes; ++lane)
                {
                    mWorkers.emplace_back(&InProcessPressureCoordinator::worker, this, lane);
                }
            }
            catch (...)
            {
                Store(&control->pressure_first_failed, 1U);
                return false;
            }
        }
        if (auto* control = mapping().get())
        {
            Store(&control->pressure_client_registered, 1U);
            FutexWake(&control->pressure_client_registered);
        }
        return true;
    }

private:
    void worker(uint32_t lane)
    {
        uint32_t previousGeneration = 0;
        while (!mStopping.load(std::memory_order_acquire))
        {
            auto* control = mapping().get();
            if (control == nullptr)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            uint32_t generation = 0;
            if (!WaitForGenerationChange(
                    &control->pressure_command_generation, previousGeneration, &generation))
            {
                continue;
            }
            if (mStopping.load(std::memory_order_acquire)) break;
            previousGeneration = generation;
            if (generation == 0 || generation != Load(&control->prepared_generation)) continue;

            auto const keyCount = Load(&control->pressure_key_count);
            auto const maxDurationMs = Load(&control->pressure_max_duration_ms);
            auto const maxLoops = Load(&control->pressure_max_loops);
            if (keyCount == 0 || maxDurationMs == 0 || maxLoops == 0)
            {
                Store(&control->pressure_first_failed, 1U);
                continue;
            }
            auto const key = pressurePrefix() + "_g" + std::to_string(generation)
                + "_pressure_" + std::to_string(lane % keyCount);
            auto const laneStarted = MonotonicNs();
            Store(&control->pressure_start_ns[lane], laneStarted);
            FetchAdd(&control->pressure_started_lanes, 1U);
            FutexWake(&control->pressure_started_lanes);
            uint32_t loops = 0;
            uint32_t errors = 0;
            while (!mStopping.load(std::memory_order_acquire)
                && Load(&control->pressure_stop_generation) != generation
                && loops < maxLoops
                && MonotonicNs() - laneStarted
                    < static_cast<uint64_t>(maxDurationMs) * 1000000ULL)
            {
                FetchAdd(&control->pressure_active_gets, 1U);
                maybeMarkEstablished(control, generation);
                bool ok = false;
                try
                {
                    ok = (*mGet)(lane, key);
                }
                catch (...)
                {
                    ok = false;
                }
                FetchSub(&control->pressure_active_gets, 1U);
                ++loops;
                if (!ok) ++errors;
                Store(&control->pressure_end_ns[lane], MonotonicNs());
                Store(&control->pressure_loop_count[lane], loops);
                Store(&control->pressure_loop_errors[lane], errors);
            }
            auto const ok = loops > 0 && errors == 0;
            Store(&control->pressure_ok[lane], ok ? 1U : 0U);
            FetchAdd(ok ? &control->pressure_success : &control->pressure_errors, 1U);
            auto const completed = FetchAdd(&control->completed_pressure_lanes, 1U) + 1U;
            FutexWake(&control->completed_pressure_lanes);
            if (completed == Load(&control->pressure_lanes))
            {
                Store(&control->pressure_stopped_generation, generation);
                FutexWake(&control->pressure_stopped_generation);
            }
        }
    }

    static void maybeMarkEstablished(SharedControl* control, uint32_t generation)
    {
        auto const lanes = Load(&control->pressure_lanes);
        auto const required = RequiredPressureFirst(lanes);
        if (Load(&control->pressure_started_lanes) == lanes
            && Load(&control->pressure_active_gets) >= required)
        {
            Store(&control->pressure_established_generation, generation);
            FutexWake(&control->pressure_established_generation);
        }
    }

    std::mutex mMutex;
    std::shared_ptr<InProcessPressureGet> mGet;
    std::vector<std::thread> mWorkers;
    std::atomic<bool> mStopping{false};
};

InProcessPressureCoordinator& inProcessCoordinator()
{
    static InProcessPressureCoordinator value;
    return value;
}

struct InProcessPressureSession::Impl
{
};

InProcessPressureSession::InProcessPressureSession() = default;

InProcessPressureSession::InProcessPressureSession(std::unique_ptr<Impl> impl)
    : mImpl(std::move(impl))
{
}

InProcessPressureSession::~InProcessPressureSession()
{
}

InProcessPressureSession::InProcessPressureSession(InProcessPressureSession&&) noexcept = default;

InProcessPressureSession& InProcessPressureSession::operator=(InProcessPressureSession&&) noexcept = default;

bool InProcessPressureSession::active() const
{
    return mImpl != nullptr;
}

void InProcessPressureSession::finish()
{
    mImpl.reset();
}

BusinessGetToken beginBusinessGet(std::string const& requestId, BusinessApi api, uint32_t keyCount)
{
    auto const callStarted = MonotonicNs();
    BusinessGetToken token;
    token.api = api;
    token.keyCount = keyCount;
    if (!kvcBurstEnabled())
    {
        token.status = TriggerStatus::kSkippedDisabled;
        token.triggerUs = (MonotonicNs() - callStarted) / 1000ULL;
        if (measureDisabled())
        {
            std::cout << "{\"event\":\"kvc_proxy_disabled_overhead\",\"request_id\":\""
                      << jsonEscape(requestId) << "\",\"overhead_us\":" << token.triggerUs << "}"
                      << std::endl;
        }
        return token;
    }
    if (requestId.empty())
    {
        token.status = TriggerStatus::kSkippedNoRequest;
        emitSkipped(requestId, token.status);
        return token;
    }
    std::unique_lock<std::mutex> claimLock(claimMutex());
    if (seenRequests().contains(requestId))
    {
        token.status = TriggerStatus::kSkippedAlreadyTriggered;
        emitSkipped(requestId, token.status);
        return token;
    }
    auto* control = mapping().get();
    if (control == nullptr)
    {
        token.status = TriggerStatus::kInvalidControl;
        emitSkipped(requestId, token.status);
        return token;
    }
    if (Load(&control->trigger_armed) == 0)
    {
        token.status = TriggerStatus::kSkippedDisarmed;
        emitSkipped(requestId, token.status);
        return token;
    }
    auto const inProcess = inProcessPressureEnabled();
    auto const pressureLanes = Load(&control->pressure_lanes);
    auto const pressureKeyCount = Load(&control->pressure_key_count);
    auto const inProcessShapeValid = Load(&control->configured_concurrency) == 32U
        && pressureLanes == 31U && pressureKeyCount > 0U && pressureKeyCount <= pressureLanes
        && Load(&control->object_size_bytes) == 3670016ULL;
    if (Load(&control->keys_verified) != pressureLanes
        || (inProcess ? !inProcessShapeValid : Load(&control->clients_connected) != pressureLanes))
    {
        token.status = TriggerStatus::kInvalidControl;
        emitSkipped(requestId, token.status);
        return token;
    }
    if (inProcess
        && (Load(&control->pressure_client_registered) == 0U
            || Load(&control->pressure_established_generation)
                != Load(&control->prepared_generation)
            || Load(&control->completed_pressure_lanes) != 0U))
    {
        token.status = TriggerStatus::kInvalidControl;
        emitSkipped(requestId, token.status);
        return token;
    }
    if (Load(&control->state) != static_cast<uint32_t>(State::kReady))
    {
        token.status = TriggerStatus::kSkippedBusy;
        emitSkipped(requestId, token.status);
        return token;
    }
    if (inProcess)
    {
        auto const waitStarted = MonotonicNs();
        auto const required = RequiredPressureFirst(pressureLanes);
        auto const deadline = DeadlineNs(Load(&control->barrier_timeout_ms));
        while (Load(&control->pressure_active_gets) < required
            && Load(&control->completed_pressure_lanes) == 0U && MonotonicNs() < deadline)
        {
            std::this_thread::yield();
        }
        token.pressureWaitUs = (MonotonicNs() - waitStarted) / 1000ULL;
        auto const active = Load(&control->pressure_active_gets);
        Store(&control->business_pressure_wait_us, token.pressureWaitUs);
        Store(&control->business_pressure_active_snapshot, active);
        if (active < required)
        {
            token.status = TriggerStatus::kInvalidControl;
            emitSkipped(requestId, token.status);
            return token;
        }
    }

    uint32_t expected = static_cast<uint32_t>(State::kReady);
    auto const triggerStarted = MonotonicNs();
    Store(&control->claim_started_ns, triggerStarted);
    if (!CompareExchange(&control->state, &expected, static_cast<uint32_t>(State::kClaimed)))
    {
        token.status = TriggerStatus::kSkippedBusy;
        emitSkipped(requestId, token.status);
        return token;
    }

    token.status = TriggerStatus::kTriggered;
    token.inProcessPressure = inProcess;
    seenRequests().insert(requestId);
    token.generation = Load(&control->prepared_generation);
    token.bytes = static_cast<uint64_t>(keyCount) * Load(&control->object_size_bytes);
    CopyRequestId(control->request_id, requestId.c_str());
    Store(&control->business_api, static_cast<uint32_t>(api));
    Store(&control->business_key_count, keyCount);
    Store(&control->business_bytes, token.bytes);
    Store(&control->trigger_started_ns, triggerStarted);
    Store(&control->business_trigger_status, static_cast<uint32_t>(TriggerStatus::kTriggered));
    Store(&control->state, static_cast<uint32_t>(State::kRunning));
    if (!token.inProcessPressure)
    {
        Store(&control->run_generation, token.generation);
        FutexWake(&control->run_generation);
    }
    auto const triggerCompleted = MonotonicNs();
    token.triggerUs = (triggerCompleted - triggerStarted) / 1000ULL;
    Store(&control->trigger_completed_ns, triggerCompleted);
    Store(&control->business_trigger_us, token.triggerUs);
    claimLock.unlock();

    if (token.inProcessPressure)
    {
        token.pressureEstablished = true;
        token.businessStartedNs = MonotonicNs();
        Store(&control->business_start_ns, token.businessStartedNs);
        return token;
    }

    token.barrierReleased = ArriveAndWait(
        control, token.generation, Load(&control->barrier_timeout_ms), &token.barrierWaitUs);
    Store(&control->business_barrier_wait_us, token.barrierWaitUs);
    token.pressureEstablished = token.barrierReleased && WaitForPressureStarted(
        control, token.generation, Load(&control->barrier_timeout_ms), &token.pressureWaitUs);
    Store(&control->business_pressure_wait_us, token.pressureWaitUs);
    if (token.pressureEstablished)
    {
        if (Load(&control->pressure_lanes) > 0)
        {
            token.leadWaitUs = ApplyPressureLead(control);
        }
        Store(&control->business_release_generation, token.generation);
        FutexWake(&control->business_release_generation);
    }
    Store(&control->business_lead_wait_us, token.leadWaitUs);
    token.businessStartedNs = MonotonicNs();
    Store(&control->business_start_ns, token.businessStartedNs);
    return token;
}

InProcessPressureSession beginInProcessPressure(
    BusinessGetToken const& token, InProcessPressureGet get)
{
    // Register only the production single-key onboard client. Registering the
    // first MGet/parallel client observed during prime would recreate the
    // original client/socket isolation ambiguity.
    if (token.api != BusinessApi::kGet) return {};
    auto const registered = inProcessCoordinator().registerClient(std::move(get));
    if (!registered) return {};
    return InProcessPressureSession(
        token.inProcessPressure ? std::make_unique<InProcessPressureSession::Impl>() : nullptr);
}

void finishBusinessGet(BusinessGetToken const& token, bool success, uint64_t businessEndedNs)
{
    if (!token.triggered()) return;
    auto* control = mapping().get();
    if (control == nullptr || token.generation != Load(&control->run_generation)) return;
    auto const ended = businessEndedNs == 0 ? MonotonicNs() : businessEndedNs;
    Store(&control->business_end_ns, ended);
    Store(&control->business_success, success ? 1U : 0U);
    Store(&control->business_done_generation, token.generation);
    FutexWake(&control->business_done_generation);
}

} // namespace pairec::kvc_burst
