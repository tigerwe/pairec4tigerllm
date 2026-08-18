#pragma once

#include "kvc_burst_shared.h"

#include <cstdint>
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
    bool barrierReleased{false};
    bool pressureEstablished{false};

    [[nodiscard]] bool triggered() const { return status == TriggerStatus::kTriggered; }
};

[[nodiscard]] bool kvcBurstEnabled();
[[nodiscard]] BusinessGetToken beginBusinessGet(
    std::string const& requestId, BusinessApi api, uint32_t keyCount);
void finishBusinessGet(BusinessGetToken const& token, bool success, uint64_t businessEndedNs = 0);

} // namespace pairec::kvc_burst
