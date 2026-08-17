#pragma once

#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pairec::kvc_burst
{

constexpr uint64_t kMagic = 0x5041495245434b56ULL;
constexpr uint32_t kVersion = 2;
constexpr uint32_t kMaxConcurrency = 100;
constexpr uint32_t kMaxPressureLanes = kMaxConcurrency - 1;
constexpr size_t kRequestIdSize = 128;

enum class State : uint32_t
{
    kStarting = 0,
    kReady = 1,
    kClaimed = 2,
    kRunning = 3,
    kFailed = 4,
    kStopping = 5,
};

enum class Failure : uint32_t
{
    kNone = 0,
    kBarrierTimeout = 1,
    kPressureGetFailed = 2,
    kBusinessGetFailed = 3,
    kInvalidControl = 4,
};

enum class BusinessApi : uint32_t
{
    kUnknown = 0,
    kGet = 1,
    kMGet = 2,
    kParallelGet = 3,
};

enum class TriggerStatus : uint32_t
{
    kNone = 0,
    kTriggered = 1,
    kSkippedDisabled = 2,
    kSkippedBusy = 3,
    kSkippedNoRequest = 4,
    kInvalidControl = 5,
    kSkippedAlreadyTriggered = 6,
    kSkippedDisarmed = 7,
};

// All fields use fixed-width integral types so the control block has one ABI in
// both containers. Shared synchronization uses __atomic builtins, not
// process-local std::atomic objects.
struct alignas(64) SharedControl
{
    uint64_t magic;
    uint32_t version;
    uint32_t struct_size;

    uint32_t configured_concurrency;
    uint32_t pressure_lanes;
    uint32_t barrier_timeout_ms;
    uint32_t state;

    uint32_t prepared_generation;
    uint32_t run_generation;
    uint32_t release_generation;
    uint32_t result_generation;

    uint32_t arrived_participants;
    uint32_t armed_pressure_lanes;
    uint32_t completed_pressure_lanes;
    uint32_t barrier_failed;

    uint32_t business_done_generation;
    uint32_t business_success;
    uint32_t pressure_success;
    uint32_t pressure_errors;

    uint32_t result_valid;
    uint32_t result_failure;
    uint32_t result_max_active_all;
    uint32_t result_business_overlap;

    uint32_t business_api;
    uint32_t business_key_count;
    uint32_t business_trigger_status;
    uint32_t keys_verified;

    uint32_t clients_connected;
    uint32_t trigger_armed;
    uint32_t reserved_u32_1;
    uint32_t reserved_u32_2;

    uint64_t object_size_bytes;
    uint64_t shuffle_seed;
    uint64_t heartbeat_ns;
    uint64_t claim_started_ns;
    uint64_t business_barrier_wait_us;

    uint64_t business_trigger_us;
    uint64_t business_bytes;
    uint64_t trigger_started_ns;
    uint64_t trigger_completed_ns;

    uint64_t business_start_ns;
    uint64_t business_end_ns;
    uint64_t result_start_skew_us;
    uint64_t result_business_get_us;

    uint64_t result_pressure_avg_us;
    uint64_t result_pressure_p95_us;
    uint64_t result_pressure_p99_us;
    uint64_t result_pressure_max_us;

    char request_id[kRequestIdSize];

    uint64_t pressure_start_ns[kMaxPressureLanes];
    uint64_t pressure_end_ns[kMaxPressureLanes];
    uint32_t pressure_ok[kMaxPressureLanes];
};

static_assert(sizeof(SharedControl) < 16384, "shared KVC burst control block unexpectedly large");

template <typename T>
inline T Load(const T* value, int order = __ATOMIC_ACQUIRE)
{
    return __atomic_load_n(value, order);
}

template <typename T>
inline void Store(T* target, T value, int order = __ATOMIC_RELEASE)
{
    __atomic_store_n(target, value, order);
}

template <typename T>
inline T FetchAdd(T* target, T value, int order = __ATOMIC_ACQ_REL)
{
    return __atomic_fetch_add(target, value, order);
}

template <typename T>
inline T FetchSub(T* target, T value, int order = __ATOMIC_ACQ_REL)
{
    return __atomic_fetch_sub(target, value, order);
}

inline bool CompareExchange(uint32_t* target, uint32_t* expected, uint32_t desired)
{
    return __atomic_compare_exchange_n(
        target, expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

inline uint64_t MonotonicNs()
{
    timespec value{};
    ::clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + static_cast<uint64_t>(value.tv_nsec);
}

inline uint64_t DeadlineNs(uint32_t timeoutMs)
{
    return MonotonicNs() + static_cast<uint64_t>(timeoutMs) * 1000000ULL;
}

inline int FutexWait(uint32_t* address, uint32_t expected, const timespec* timeout)
{
    return static_cast<int>(::syscall(SYS_futex, address, FUTEX_WAIT, expected, timeout, nullptr, 0));
}

inline int FutexWake(uint32_t* address, int count = INT_MAX)
{
    return static_cast<int>(::syscall(SYS_futex, address, FUTEX_WAKE, count, nullptr, nullptr, 0));
}

inline bool WaitForValue(uint32_t* address, uint32_t expectedValue, uint64_t deadlineNs)
{
    while (Load(address) != expectedValue)
    {
        auto now = MonotonicNs();
        if (now >= deadlineNs)
        {
            return Load(address) == expectedValue;
        }
        auto remaining = deadlineNs - now;
        timespec timeout{static_cast<time_t>(remaining / 1000000000ULL),
            static_cast<long>(remaining % 1000000000ULL)};
        auto observed = Load(address);
        if (observed == expectedValue)
        {
            return true;
        }
        auto status = FutexWait(address, observed, &timeout);
        if (status != 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT)
        {
            return false;
        }
    }
    return true;
}

inline bool WaitForGenerationChange(uint32_t* address, uint32_t previous, uint32_t* observed)
{
    while ((*observed = Load(address)) == previous)
    {
        auto status = FutexWait(address, previous, nullptr);
        if (status != 0 && errno != EAGAIN && errno != EINTR)
        {
            return false;
        }
    }
    return true;
}

inline bool ArriveAndWait(SharedControl* control, uint32_t generation, uint32_t timeoutMs, uint64_t* waitUs)
{
    auto started = MonotonicNs();
    auto participants = Load(&control->configured_concurrency);
    auto arrived = FetchAdd(&control->arrived_participants, 1U) + 1U;
    if (arrived == participants)
    {
        Store(&control->release_generation, generation);
        FutexWake(&control->release_generation);
    }

    auto deadline = started + static_cast<uint64_t>(timeoutMs) * 1000000ULL;
    auto released = WaitForValue(&control->release_generation, generation, deadline);
    if (!released)
    {
        // A timeout in any participant invalidates this generation and wakes
        // every waiter. Business code may then continue its real Get.
        Store(&control->barrier_failed, 1U);
        Store(&control->release_generation, generation);
        FutexWake(&control->release_generation);
    }
    if (waitUs != nullptr)
    {
        *waitUs = (MonotonicNs() - started) / 1000ULL;
    }
    return released && Load(&control->barrier_failed) == 0;
}

inline bool IsCompatible(const SharedControl& control)
{
    return control.magic == kMagic && control.version == kVersion && control.struct_size == sizeof(SharedControl)
        && control.configured_concurrency >= 1 && control.configured_concurrency <= kMaxConcurrency
        && control.pressure_lanes + 1 == control.configured_concurrency;
}

inline const char* BusinessApiName(BusinessApi api)
{
    switch (api)
    {
    case BusinessApi::kGet: return "get";
    case BusinessApi::kMGet: return "mget";
    case BusinessApi::kParallelGet: return "parallel_get";
    default: return "unknown";
    }
}

inline void CopyRequestId(char* destination, const char* source)
{
    std::memset(destination, 0, kRequestIdSize);
    if (source != nullptr)
    {
        std::strncpy(destination, source, kRequestIdSize - 1);
    }
}

} // namespace pairec::kvc_burst
