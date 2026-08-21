#include "kvc_operation_proxy.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
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
    return "PAIREC_KVC_BURST_PROXY_V4";
}

extern "C" char const* pairecKvcInProcessBurstCapabilityMarker()
{
    return "PAIREC_KVC_INPROCESS_BURST_C32_V1";
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

struct InProcessPressureSession::Impl
{
    std::mutex mutex;
    std::condition_variable readyCondition;
    std::condition_variable launchCondition;
    std::vector<std::thread> threads;
    uint32_t ready{0};
    bool launch{false};

    void finish()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            launch = true;
        }
        launchCondition.notify_all();
        for (auto& thread : threads)
        {
            if (thread.joinable()) thread.join();
        }
        threads.clear();
    }
};

InProcessPressureSession::InProcessPressureSession() = default;

InProcessPressureSession::InProcessPressureSession(std::unique_ptr<Impl> impl)
    : mImpl(std::move(impl))
{
}

InProcessPressureSession::~InProcessPressureSession()
{
    finish();
}

InProcessPressureSession::InProcessPressureSession(InProcessPressureSession&&) noexcept = default;

InProcessPressureSession& InProcessPressureSession::operator=(InProcessPressureSession&&) noexcept = default;

bool InProcessPressureSession::active() const
{
    return mImpl != nullptr;
}

void InProcessPressureSession::finish()
{
    if (mImpl == nullptr) return;
    mImpl->finish();
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
    if (Load(&control->state) != static_cast<uint32_t>(State::kReady))
    {
        token.status = TriggerStatus::kSkippedBusy;
        emitSkipped(requestId, token.status);
        return token;
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
    Store(&control->run_generation, token.generation);
    FutexWake(&control->run_generation);
    auto const triggerCompleted = MonotonicNs();
    token.triggerUs = (triggerCompleted - triggerStarted) / 1000ULL;
    Store(&control->trigger_completed_ns, triggerCompleted);
    Store(&control->business_trigger_us, token.triggerUs);
    claimLock.unlock();

    if (token.inProcessPressure)
    {
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
    if (!token.triggered() || !token.inProcessPressure || !get)
    {
        return {};
    }
    auto* control = mapping().get();
    if (control == nullptr || token.generation != Load(&control->run_generation))
    {
        return {};
    }

    auto const lanes = Load(&control->pressure_lanes);
    auto const keyCount = Load(&control->pressure_key_count);
    if (keyCount == 0U || keyCount > lanes)
    {
        Store(&control->pressure_first_failed, 1U);
        return {};
    }
    auto impl = std::make_unique<InProcessPressureSession::Impl>();
    auto* state = impl.get();
    auto getShared = std::make_shared<InProcessPressureGet>(std::move(get));
    auto const prefix = pressurePrefix() + "_g" + std::to_string(token.generation) + "_pressure_";
    uint32_t created = 0;
    try
    {
        impl->threads.reserve(lanes);
        for (uint32_t lane = 0; lane < lanes; ++lane)
        {
            impl->threads.emplace_back([state, control, getShared, prefix, lane, keyCount] {
                {
                    std::unique_lock<std::mutex> lock(state->mutex);
                    ++state->ready;
                    state->readyCondition.notify_one();
                    state->launchCondition.wait(lock, [state] { return state->launch; });
                }
                auto const started = MonotonicNs();
                Store(&control->pressure_start_ns[lane], started);
                FetchAdd(&control->pressure_started_lanes, 1U);
                FutexWake(&control->pressure_started_lanes);
                bool ok = false;
                try
                {
                    ok = (*getShared)(lane, prefix + std::to_string(lane % keyCount));
                }
                catch (...)
                {
                    ok = false;
                }
                Store(&control->pressure_end_ns[lane], MonotonicNs());
                Store(&control->pressure_ok[lane], ok ? 1U : 0U);
                FetchAdd(ok ? &control->pressure_success : &control->pressure_errors, 1U);
                FetchAdd(&control->completed_pressure_lanes, 1U);
                FutexWake(&control->completed_pressure_lanes);
            });
            ++created;
        }
    }
    catch (...)
    {
        Store(&control->pressure_first_failed, 1U);
    }

    auto const deadline = DeadlineNs(Load(&control->barrier_timeout_ms));
    {
        std::unique_lock<std::mutex> lock(impl->mutex);
        while (impl->ready != created && MonotonicNs() < deadline)
        {
            impl->readyCondition.wait_for(lock, std::chrono::milliseconds(1));
        }
        impl->launch = true;
    }
    impl->launchCondition.notify_all();

    while (Load(&control->pressure_started_lanes) < created
        && Load(&control->completed_pressure_lanes) == 0U && MonotonicNs() < deadline)
    {
        std::this_thread::yield();
    }
    auto const started = Load(&control->pressure_started_lanes);
    auto const completed = Load(&control->completed_pressure_lanes);
    auto const established = created == lanes && started == lanes && completed == 0U;
    if (!established)
    {
        Store(&control->pressure_first_failed, 1U);
    }
    for (uint32_t lane = created; lane < lanes; ++lane)
    {
        auto const now = MonotonicNs();
        Store(&control->pressure_start_ns[lane], now);
        Store(&control->pressure_end_ns[lane], now);
        Store(&control->pressure_ok[lane], 0U);
        FetchAdd(&control->pressure_errors, 1U);
        FetchAdd(&control->completed_pressure_lanes, 1U);
    }
    Store(&control->business_barrier_wait_us, uint64_t{0});
    Store(&control->business_pressure_wait_us, uint64_t{0});
    Store(&control->business_lead_wait_us, uint64_t{0});
    Store(&control->business_release_generation, token.generation);
    FutexWake(&control->business_release_generation);
    Store(&control->business_start_ns, MonotonicNs());
    return InProcessPressureSession(std::move(impl));
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
