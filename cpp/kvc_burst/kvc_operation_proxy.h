#pragma once

#include "kvc_burst_shared.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace pairec::kvc_burst
{

struct BusinessGetToken
{
    TriggerStatus status{TriggerStatus::kNone};
    uint32_t generation{0};
    BusinessApi api{BusinessApi::kUnknown};
    uint32_t keyCount{0};
    uint64_t bytes{0};
    uint64_t triggerUs{0};
    uint64_t barrierWaitUs{0};
    uint64_t pressureWaitUs{0};
    uint64_t leadWaitUs{0};
    uint64_t businessStartedNs{0};
    uint32_t businessGetOrdinal{0};
    bool barrierReleased{false};
    bool pressureEstablished{false};
    bool inProcessPressure{false};

    [[nodiscard]] bool triggered() const { return status == TriggerStatus::kTriggered; }
};

using InProcessPressureGet = std::function<bool(uint32_t, std::string const&)>;

class InProcessPressureSession
{
public:
    InProcessPressureSession();
    ~InProcessPressureSession();
    InProcessPressureSession(InProcessPressureSession&&) noexcept;
    InProcessPressureSession& operator=(InProcessPressureSession&&) noexcept;
    InProcessPressureSession(InProcessPressureSession const&) = delete;
    InProcessPressureSession& operator=(InProcessPressureSession const&) = delete;

    [[nodiscard]] bool active() const;
    void finish();

private:
    struct Impl;
    explicit InProcessPressureSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> mImpl;

    friend InProcessPressureSession beginInProcessPressure(
        BusinessGetToken const&, InProcessPressureGet);
};

[[nodiscard]] bool kvcBurstEnabled();
[[nodiscard]] BusinessGetToken beginBusinessGet(
    std::string const& requestId, BusinessApi api, uint32_t keyCount);
[[nodiscard]] InProcessPressureSession beginInProcessPressure(
    BusinessGetToken const& token, InProcessPressureGet get);
[[nodiscard]] bool registerInProcessPressureClient(InProcessPressureGet get);
void shutdownInProcessPressureClient();
void finishBusinessGet(BusinessGetToken const& token, bool success, uint64_t businessEndedNs = 0);
void observeAddTokenStart(std::string const& requestId);
void observeAddTokenEnd(std::string const& requestId);
void observeBusinessRequestComplete(std::string const& requestId);

} // namespace pairec::kvc_burst
