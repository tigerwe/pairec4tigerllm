#!/usr/bin/env python3
"""Patch the native TensorRT-LLM DataSystem path with exact request attribution."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_V1"


TRACKER_HEADER = r'''/* PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_V1 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

enum class DataSystemOperation
{
    kGet,
    kSet,
};

enum class DataSystemRequestPhase
{
    kAddSequence,
    kAddToken,
    kRemoveSequence,
};

class DataSystemRequestScope
{
public:
    explicit DataSystemRequestScope(std::optional<std::uint64_t> correlationId);
    ~DataSystemRequestScope();

    DataSystemRequestScope(DataSystemRequestScope const&) = delete;
    DataSystemRequestScope& operator=(DataSystemRequestScope const&) = delete;

private:
    bool mActive{false};
    std::optional<std::uint64_t> mPrevious;
};

[[nodiscard]] bool dataSystemRequestAttributionEnabled();
[[nodiscard]] std::optional<std::string> currentDataSystemRequestId();
void registerDataSystemRequest(
    std::uint64_t correlationId, std::string const& requestId, std::uint64_t nativeLifecycleCount);
void finishDataSystemRequest(std::uint64_t correlationId);
void markDataSystemRequestExecutorEnqueued(std::uint64_t correlationId);
void recordDataSystemRequestPhase(std::optional<std::uint64_t> correlationId, DataSystemRequestPhase phase,
    std::int64_t durationUs, std::int64_t attributionLookupUs = 0, std::int64_t sequenceLookupUs = 0,
    std::int64_t bodyUs = 0);
[[nodiscard]] std::uint64_t beginDataSystemOperation(DataSystemOperation operation);
void finishDataSystemOperation(
    std::uint64_t token, DataSystemOperation operation, std::int64_t durationUs, bool failed);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
'''


TRACKER_SOURCE = r'''/* PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_V1 */
#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
namespace
{

using Clock = std::chrono::steady_clock;

struct RequestStats
{
    std::string requestId;
    Clock::time_point registeredAt{Clock::now()};
    std::int64_t getUs{0};
    std::int64_t setUs{0};
    std::uint64_t getCount{0};
    std::uint64_t setCount{0};
    std::uint64_t getFailedCount{0};
    std::uint64_t setFailedCount{0};
    std::uint64_t pendingCount{0};
    std::uint64_t unknownCount{0};
    std::uint64_t nativeLifecyclesPending{1};
    std::optional<Clock::time_point> executorEnqueuedAt;
    std::optional<Clock::time_point> nativeLifecycleStartedAt;
    std::optional<Clock::time_point> previousPhaseEndedAt;
    std::uint64_t currentAddTokenCount{0};
    std::int64_t executorQueueUs{0};
    std::int64_t addSequenceUs{0};
    std::uint64_t addSequenceCount{0};
    std::int64_t prefillGapUs{0};
    std::int64_t addTokenUs{0};
    std::uint64_t addTokenCount{0};
    std::int64_t attributionLookupUs{0};
    std::int64_t sequenceLookupUs{0};
    std::int64_t kvUpdateUs{0};
    std::int64_t phaseRecordUs{0};
    std::int64_t decodeGapUs{0};
    std::int64_t finalizationGapUs{0};
    std::int64_t removeSequenceUs{0};
    std::uint64_t removeSequenceCount{0};
    std::int64_t nativeLifecycleUs{0};
    std::uint64_t nativeLifecycleCount{0};
    std::uint64_t phaseUnknownCount{0};
    bool lifecycleFinished{false};
    bool timedOut{false};
};

thread_local std::optional<std::uint64_t> gCorrelationId;

bool parseEnabled(char const* value)
{
    if (value == nullptr)
    {
        return false;
    }
    std::string const text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

std::chrono::seconds attributionTtl()
{
    auto const* value = std::getenv("TRTLLM_DATASYSTEM_ATTRIBUTION_TTL_SECONDS");
    auto const parsed = value == nullptr ? 30 : std::atoi(value);
    return std::chrono::seconds(std::max(parsed, 1));
}

std::string jsonEscape(std::string const& input)
{
    std::ostringstream output;
    for (unsigned char ch : input)
    {
        switch (ch)
        {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (ch < 0x20)
            {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
                       << std::dec;
            }
            else
            {
                output << ch;
            }
        }
    }
    return output.str();
}

void emitComplete(std::uint64_t correlationId, RequestStats const& stats)
{
    bool const complete = !stats.timedOut && stats.pendingCount == 0 && stats.unknownCount == 0;
    auto const nativeAccountedUs = stats.executorQueueUs + stats.addSequenceUs + stats.prefillGapUs
        + stats.addTokenUs + stats.decodeGapUs + stats.finalizationGapUs + stats.removeSequenceUs;
    auto const nativeClosureErrorUs = stats.nativeLifecycleUs >= nativeAccountedUs
        ? stats.nativeLifecycleUs - nativeAccountedUs
        : nativeAccountedUs - stats.nativeLifecycleUs;
    bool const phaseTimingComplete = stats.phaseUnknownCount == 0 && stats.nativeLifecycleCount > 0
        && stats.nativeLifecycleCount == stats.addSequenceCount
        && stats.nativeLifecycleCount == stats.removeSequenceCount && nativeClosureErrorUs <= 100;
    std::ostringstream event;
    event << "{\"event\":\"datasystem_request_complete\""
          << ",\"request_id\":\"" << jsonEscape(stats.requestId) << "\""
          << ",\"correlation_id\":" << correlationId
          << ",\"get_count\":" << stats.getCount
          << ",\"get_us\":" << stats.getUs
          << ",\"set_count\":" << stats.setCount
          << ",\"set_us\":" << stats.setUs
          << ",\"get_failed_count\":" << stats.getFailedCount
          << ",\"set_failed_count\":" << stats.setFailedCount
          << ",\"pending_count\":" << stats.pendingCount
          << ",\"unknown_count\":" << stats.unknownCount
          << ",\"executor_queue_us\":" << stats.executorQueueUs
          << ",\"add_sequence_count\":" << stats.addSequenceCount
          << ",\"add_sequence_us\":" << stats.addSequenceUs
          << ",\"prefill_gap_us\":" << stats.prefillGapUs
          << ",\"add_token_count\":" << stats.addTokenCount
          << ",\"add_token_us\":" << stats.addTokenUs
          << ",\"attribution_lookup_us\":" << stats.attributionLookupUs
          << ",\"sequence_lookup_us\":" << stats.sequenceLookupUs
          << ",\"kv_update_us\":" << stats.kvUpdateUs
          << ",\"phase_record_us\":" << stats.phaseRecordUs
          << ",\"decode_gap_us\":" << stats.decodeGapUs
          << ",\"finalization_gap_us\":" << stats.finalizationGapUs
          << ",\"remove_sequence_count\":" << stats.removeSequenceCount
          << ",\"remove_sequence_us\":" << stats.removeSequenceUs
          << ",\"native_lifecycle_count\":" << stats.nativeLifecycleCount
          << ",\"native_lifecycle_us\":" << stats.nativeLifecycleUs
          << ",\"native_accounted_us\":" << nativeAccountedUs
          << ",\"native_closure_error_us\":" << nativeClosureErrorUs
          << ",\"phase_unknown_count\":" << stats.phaseUnknownCount
          << ",\"phase_timing_complete\":" << (phaseTimingComplete ? "true" : "false")
          << ",\"attribution_complete\":" << (complete ? "true" : "false")
          << ",\"reason\":\"" << (stats.timedOut ? "attribution_timeout" : "request_lifecycle_complete")
          << "\"}";
    TLLM_LOG_INFO("%s", event.str().c_str());
}

class Tracker
{
public:
    Tracker()
        : mJanitor([this] { janitorLoop(); })
    {
        if (dataSystemRequestAttributionEnabled())
        {
            TLLM_LOG_INFO("{\"event\":\"datasystem_attribution_ready\",\"version\":3}");
        }
    }

    ~Tracker()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStopping = true;
        }
        mCondition.notify_all();
        if (mJanitor.joinable())
        {
            mJanitor.join();
        }
    }

    void registerRequest(
        std::uint64_t correlationId, std::string const& requestId, std::uint64_t nativeLifecycleCount)
    {
        if (!dataSystemRequestAttributionEnabled())
        {
            return;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto [it, inserted] = mRequests.emplace(correlationId, RequestStats{});
        if (!inserted)
        {
            it->second.unknownCount++;
        }
        it->second.requestId = requestId;
        it->second.registeredAt = Clock::now();
        it->second.nativeLifecyclesPending = std::max<std::uint64_t>(nativeLifecycleCount, 1);
    }

    void finishRequest(std::uint64_t correlationId)
    {
        std::optional<RequestStats> completed;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto it = mRequests.find(correlationId);
            if (it == mRequests.end())
            {
                return;
            }
            if (it->second.nativeLifecyclesPending == 0)
            {
                it->second.unknownCount++;
            }
            else
            {
                it->second.nativeLifecyclesPending--;
            }
            it->second.lifecycleFinished = it->second.nativeLifecyclesPending == 0;
            if (it->second.lifecycleFinished && it->second.pendingCount == 0)
            {
                completed = std::move(it->second);
                mRequests.erase(it);
            }
        }
        if (completed)
        {
            emitComplete(correlationId, *completed);
        }
    }

    void markExecutorEnqueued(std::uint64_t correlationId)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mRequests.find(correlationId);
        if (it == mRequests.end())
        {
            return;
        }
        auto& stats = it->second;
        stats.executorEnqueuedAt = Clock::now();
        stats.nativeLifecycleStartedAt.reset();
        stats.previousPhaseEndedAt.reset();
        stats.currentAddTokenCount = 0;
    }

    void recordPhase(std::uint64_t correlationId, DataSystemRequestPhase phase, std::int64_t durationUs,
        std::int64_t attributionLookupUs, std::int64_t sequenceLookupUs, std::int64_t bodyUs)
    {
        auto const recordStartedAt = Clock::now();
        auto const endedAt = Clock::now();
        auto const safeDurationUs = std::max<std::int64_t>(durationUs, 0);
        auto const startedAt = endedAt - std::chrono::microseconds(safeDurationUs);
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mRequests.find(correlationId);
        if (it == mRequests.end())
        {
            return;
        }
        auto& stats = it->second;
        if (phase == DataSystemRequestPhase::kAddSequence)
        {
            if (stats.executorEnqueuedAt)
            {
                stats.executorQueueUs += std::max<std::int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(startedAt - *stats.executorEnqueuedAt)
                        .count(),
                    0);
                stats.nativeLifecycleStartedAt = stats.executorEnqueuedAt;
            }
            else
            {
                stats.nativeLifecycleStartedAt = startedAt;
                stats.phaseUnknownCount++;
            }
            stats.addSequenceUs += safeDurationUs;
            stats.addSequenceCount++;
            stats.previousPhaseEndedAt = endedAt;
            stats.currentAddTokenCount = 0;
            stats.phaseRecordUs += std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - recordStartedAt).count();
            return;
        }
        if (phase == DataSystemRequestPhase::kAddToken)
        {
            if (stats.previousPhaseEndedAt)
            {
                auto const gapUs = std::max<std::int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(startedAt - *stats.previousPhaseEndedAt)
                        .count(),
                    0);
                if (stats.currentAddTokenCount == 0)
                {
                    stats.prefillGapUs += gapUs;
                }
                else
                {
                    stats.decodeGapUs += gapUs;
                }
            }
            else
            {
                stats.phaseUnknownCount++;
            }
            stats.addTokenUs += safeDurationUs;
            stats.addTokenCount++;
            stats.attributionLookupUs += std::max<std::int64_t>(attributionLookupUs, 0);
            stats.sequenceLookupUs += std::max<std::int64_t>(sequenceLookupUs, 0);
            stats.kvUpdateUs += std::max<std::int64_t>(bodyUs, 0);
            stats.previousPhaseEndedAt = endedAt;
            stats.currentAddTokenCount++;
            stats.phaseRecordUs += std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - recordStartedAt).count();
            return;
        }

        if (stats.previousPhaseEndedAt)
        {
            stats.finalizationGapUs += std::max<std::int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(startedAt - *stats.previousPhaseEndedAt)
                    .count(),
                0);
        }
        else
        {
            stats.phaseUnknownCount++;
        }
        stats.removeSequenceUs += safeDurationUs;
        stats.removeSequenceCount++;
        if (stats.nativeLifecycleStartedAt)
        {
            stats.nativeLifecycleUs += std::max<std::int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(endedAt - *stats.nativeLifecycleStartedAt)
                    .count(),
                0);
            stats.nativeLifecycleCount++;
        }
        else
        {
            stats.phaseUnknownCount++;
        }
        stats.executorEnqueuedAt.reset();
        stats.nativeLifecycleStartedAt.reset();
        stats.previousPhaseEndedAt.reset();
        stats.currentAddTokenCount = 0;
        stats.phaseRecordUs += std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - recordStartedAt).count();
    }

    std::uint64_t beginOperation()
    {
        if (!dataSystemRequestAttributionEnabled() || !gCorrelationId)
        {
            TLLM_LOG_WARNING(
                "{\"event\":\"datasystem_unattributed_operation\",\"reason\":\"missing_request_context\"}");
            return 0;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mRequests.find(*gCorrelationId);
        if (it == mRequests.end())
        {
            TLLM_LOG_WARNING(
                "{\"event\":\"datasystem_unattributed_operation\",\"reason\":\"unknown_correlation_id\"}");
            return 0;
        }
        it->second.pendingCount++;
        return *gCorrelationId;
    }

    std::optional<std::string> currentRequestId()
    {
        if (!gCorrelationId)
        {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto const it = mRequests.find(*gCorrelationId);
        if (it == mRequests.end())
        {
            return std::nullopt;
        }
        return it->second.requestId;
    }

    void finishOperation(std::uint64_t correlationId, DataSystemOperation operation, std::int64_t durationUs,
        bool failed)
    {
        if (correlationId == 0)
        {
            return;
        }
        std::optional<RequestStats> completed;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto it = mRequests.find(correlationId);
            if (it == mRequests.end())
            {
                return;
            }
            auto& stats = it->second;
            if (operation == DataSystemOperation::kGet)
            {
                stats.getCount++;
                stats.getUs += std::max<std::int64_t>(durationUs, 0);
                stats.getFailedCount += failed ? 1 : 0;
            }
            else
            {
                stats.setCount++;
                stats.setUs += std::max<std::int64_t>(durationUs, 0);
                stats.setFailedCount += failed ? 1 : 0;
            }
            if (stats.pendingCount == 0)
            {
                stats.unknownCount++;
            }
            else
            {
                stats.pendingCount--;
            }
            if (stats.lifecycleFinished && stats.pendingCount == 0)
            {
                completed = std::move(stats);
                mRequests.erase(it);
            }
        }
        if (completed)
        {
            emitComplete(correlationId, *completed);
        }
    }

private:
    void janitorLoop()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        while (!mStopping)
        {
            mCondition.wait_for(lock, std::chrono::seconds(1), [this] { return mStopping; });
            if (mStopping)
            {
                break;
            }
            auto const now = Clock::now();
            std::vector<std::pair<std::uint64_t, RequestStats>> expired;
            for (auto it = mRequests.begin(); it != mRequests.end();)
            {
                if (now - it->second.registeredAt >= attributionTtl())
                {
                    it->second.timedOut = true;
                    expired.emplace_back(it->first, std::move(it->second));
                    it = mRequests.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            lock.unlock();
            for (auto const& [correlationId, stats] : expired)
            {
                emitComplete(correlationId, stats);
            }
            lock.lock();
        }
    }

    std::mutex mMutex;
    std::condition_variable mCondition;
    std::unordered_map<std::uint64_t, RequestStats> mRequests;
    bool mStopping{false};
    std::thread mJanitor;
};

Tracker& tracker()
{
    static Tracker instance;
    return instance;
}

} // namespace

bool dataSystemRequestAttributionEnabled()
{
    static bool const enabled = parseEnabled(std::getenv("TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION"));
    return enabled;
}

std::optional<std::string> currentDataSystemRequestId()
{
    if (!dataSystemRequestAttributionEnabled())
    {
        return std::nullopt;
    }
    return tracker().currentRequestId();
}

DataSystemRequestScope::DataSystemRequestScope(std::optional<std::uint64_t> correlationId)
{
    mActive = dataSystemRequestAttributionEnabled();
    if (mActive)
    {
        mPrevious = gCorrelationId;
        gCorrelationId = correlationId;
    }
}

DataSystemRequestScope::~DataSystemRequestScope()
{
    if (mActive)
    {
        gCorrelationId = mPrevious;
    }
}

void registerDataSystemRequest(
    std::uint64_t correlationId, std::string const& requestId, std::uint64_t nativeLifecycleCount)
{
    if (!dataSystemRequestAttributionEnabled())
    {
        return;
    }
    tracker().registerRequest(correlationId, requestId, nativeLifecycleCount);
}

void finishDataSystemRequest(std::uint64_t correlationId)
{
    if (!dataSystemRequestAttributionEnabled())
    {
        return;
    }
    tracker().finishRequest(correlationId);
}

void markDataSystemRequestExecutorEnqueued(std::uint64_t correlationId)
{
    if (!dataSystemRequestAttributionEnabled())
    {
        return;
    }
    tracker().markExecutorEnqueued(correlationId);
}

void recordDataSystemRequestPhase(std::optional<std::uint64_t> correlationId, DataSystemRequestPhase phase,
    std::int64_t durationUs, std::int64_t attributionLookupUs, std::int64_t sequenceLookupUs, std::int64_t bodyUs)
{
    if (!dataSystemRequestAttributionEnabled() || !correlationId)
    {
        return;
    }
    tracker().recordPhase(*correlationId, phase, durationUs, attributionLookupUs, sequenceLookupUs, bodyUs);
}

std::uint64_t beginDataSystemOperation(DataSystemOperation)
{
    if (!dataSystemRequestAttributionEnabled())
    {
        return 0;
    }
    return tracker().beginOperation();
}

void finishDataSystemOperation(
    std::uint64_t token, DataSystemOperation operation, std::int64_t durationUs, bool failed)
{
    if (!dataSystemRequestAttributionEnabled())
    {
        return;
    }
    tracker().finishOperation(token, operation, durationUs, failed);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
'''


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor in {path}, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1))


def write_managed(path: Path, content: str) -> None:
    if path.exists() and MARKER not in path.read_text():
        raise RuntimeError(f"refusing to overwrite unmanaged file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def replace_supported(path: Path, alternatives: tuple[str, ...], new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    matches = [old for old in alternatives if text.count(old) == 1]
    if matches:
        longest = max(matches, key=len)
        matches = [old for old in matches if len(old) == len(longest)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one supported anchor in {path}, found {len(matches)}: "
            f"{tuple(old[:80] for old in alternatives)!r}")
    path.write_text(text.replace(matches[0], new, 1))


def patch_parallel_get(transfer: Path) -> None:
    transfer_text = transfer.read_text()
    if ("TRTLLM_DATASYSTEM_PARALLEL_GET" not in transfer_text
            or "attributionParallelGetToken" in transfer_text):
        return
    replace_once(transfer,
        "                for (auto const& key : keys)\n"
        "                {\n"
        "                    futures.emplace_back(dataSystemGetExecutor().submit("
        "[kvClient, key, monotonicMs]() mutable {\n"
        "                        DataSystemGetResult result;\n"
        "                        auto const started = monotonicMs();\n",
        "                for (auto const& key : keys)\n"
        "                {\n"
        "                    auto const attributionParallelGetToken\n"
        "                        = beginDataSystemOperation(DataSystemOperation::kGet);\n"
        "                    futures.emplace_back(dataSystemGetExecutor().submit(\n"
        "                        [kvClient, key, monotonicMs, attributionParallelGetToken]() mutable {\n"
        "                        DataSystemGetResult result;\n"
        "                        auto const started = monotonicMs();\n"
        "                        auto const attributionParallelGetStarted = std::chrono::steady_clock::now();\n")
    replace_once(transfer,
        "                        result.getMs = monotonicMs() - started;\n"
        "                        return result;\n",
        "                        result.getMs = monotonicMs() - started;\n"
        "                        auto const attributionParallelGetUs\n"
        "                            = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "                                std::chrono::steady_clock::now()\n"
        "                                - attributionParallelGetStarted)\n"
        "                                  .count();\n"
        "                        finishDataSystemOperation(attributionParallelGetToken,\n"
        "                            DataSystemOperation::kGet, attributionParallelGetUs, !result.ok);\n"
        "                        return result;\n")


def patch_tree(root: Path) -> None:
    include_dir = root / "cpp/include/tensorrt_llm/batch_manager"
    source_dir = root / "cpp/tensorrt_llm/batch_manager"
    required = [source_dir / "CMakeLists.txt", source_dir / "kvCacheManager.cpp",
                source_dir / "kvCacheTransferManager.cpp", include_dir / "kvCacheManager.h"]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"missing TensorRT-LLM source file: {path}")

    write_managed(include_dir / "datasystemRequestTracker.h", TRACKER_HEADER)
    write_managed(source_dir / "datasystemRequestTracker.cpp", TRACKER_SOURCE)

    cmake = source_dir / "CMakeLists.txt"
    replace_once(cmake, "    dataTransceiverImpl.cpp\n", "    dataTransceiverImpl.cpp\n    datasystemRequestTracker.cpp\n")

    manager_h = include_dir / "kvCacheManager.h"
    old_client_id_map = (
        "    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;\n"
        "    // Exact Executor clientId used only for DataSystem request attribution.\n"
        "    std::unordered_map<LlmRequest::RequestIdType, std::optional<LlmRequest::RequestIdType>>\n"
        "        mDataSystemClientIds;\n")
    new_client_id_map = (
        "    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;\n"
        "    // Keep attribution lookups out of the scheduler's sequence critical section.\n"
        "    std::mutex mDataSystemClientIdsMtx;\n"
        "    std::unordered_map<LlmRequest::RequestIdType, std::optional<LlmRequest::RequestIdType>>\n"
        "        mDataSystemClientIds;\n")
    replace_supported(manager_h, (
        "    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;\n",
        old_client_id_map,
    ), new_client_id_map)

    manager = source_dir / "kvCacheManager.cpp"
    replace_once(manager,
        '#include "tensorrt_llm/batch_manager/kvCacheManager.h"\n',
        '#include "tensorrt_llm/batch_manager/kvCacheManager.h"\n'
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n')
    manager_text = manager.read_text()
    if "#include <chrono>" not in manager_text:
        if "#include <algorithm>\n" in manager_text:
            replace_once(manager, "#include <algorithm>\n", "#include <algorithm>\n#include <chrono>\n")
        else:
            replace_once(manager,
                '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n',
                '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n#include <chrono>\n')

    original_add_token = (
        "void KVCacheManager::addToken(RequestIdType requestId)\n{\n"
        "    auto& sequence = getSequence(requestId);\n"
        "    updateToken(sequence, true);\n"
        "}\n")
    old_add_token = (
        "void KVCacheManager::addToken(RequestIdType requestId)\n{\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    auto& sequence = getSequence(requestId);\n"
        "    updateToken(sequence, true);\n"
        "}\n")
    v2_add_token = (
        "void KVCacheManager::addToken(RequestIdType requestId)\n{\n"
        "    // PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    if (dataSystemRequestAttributionEnabled())\n"
        "    {\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    auto& sequence = getSequence(requestId);\n"
        "    updateToken(sequence, true);\n"
        "}\n")
    legacy_v3_add_token = (
        "void KVCacheManager::addToken(RequestIdType requestId)\n{\n"
        "    // PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2\n"
        "    // PAIREC_TRT_EXECUTOR_PHASE_TIMING_V3\n"
        "    auto const attributionAddTokenStarted = std::chrono::steady_clock::now();\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    std::int64_t attributionLookupUs = 0;\n"
        "    if (dataSystemRequestAttributionEnabled())\n"
        "    {\n"
        "        auto const attributionLookupStarted = std::chrono::steady_clock::now();\n"
        "        {\n"
        "            std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "            auto const it = mDataSystemClientIds.find(requestId);\n"
        "            if (it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "        }\n"
        "        attributionLookupUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "            std::chrono::steady_clock::now() - attributionLookupStarted).count();\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    auto const attributionSequenceLookupStarted = std::chrono::steady_clock::now();\n"
        "    auto& sequence = getSequence(requestId);\n"
        "    auto const attributionSequenceLookupUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "        std::chrono::steady_clock::now() - attributionSequenceLookupStarted).count();\n"
        "    auto const attributionKvUpdateStarted = std::chrono::steady_clock::now();\n"
        "    updateToken(sequence, true);\n"
        "    auto const attributionKvUpdateUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "        std::chrono::steady_clock::now() - attributionKvUpdateStarted).count();\n"
        "    auto const attributionAddTokenUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "        std::chrono::steady_clock::now() - attributionAddTokenStarted).count();\n"
        "    recordDataSystemRequestPhase(clientId, DataSystemRequestPhase::kAddToken, attributionAddTokenUs,\n"
        "        attributionLookupUs, attributionSequenceLookupUs, attributionKvUpdateUs);\n"
        "}\n")
    new_add_token = (
        "void KVCacheManager::addToken(RequestIdType requestId)\n{\n"
        "    // PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2\n"
        "    // PAIREC_TRT_EXECUTOR_PHASE_TIMING_V3\n"
        "    auto const attributionEnabled = dataSystemRequestAttributionEnabled();\n"
        "    auto const attributionAddTokenStarted = attributionEnabled\n"
        "        ? std::chrono::steady_clock::now()\n"
        "        : std::chrono::steady_clock::time_point{};\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    std::int64_t attributionLookupUs = 0;\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        auto const attributionLookupStarted = std::chrono::steady_clock::now();\n"
        "        {\n"
        "            std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "            auto const it = mDataSystemClientIds.find(requestId);\n"
        "            if (it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "        }\n"
        "        attributionLookupUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "            std::chrono::steady_clock::now() - attributionLookupStarted).count();\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    auto const attributionSequenceLookupStarted = attributionEnabled\n"
        "        ? std::chrono::steady_clock::now()\n"
        "        : std::chrono::steady_clock::time_point{};\n"
        "    auto& sequence = getSequence(requestId);\n"
        "    auto const attributionSequenceLookupUs = attributionEnabled\n"
        "        ? std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "              std::chrono::steady_clock::now() - attributionSequenceLookupStarted).count()\n"
        "        : 0;\n"
        "    auto const attributionKvUpdateStarted = attributionEnabled\n"
        "        ? std::chrono::steady_clock::now()\n"
        "        : std::chrono::steady_clock::time_point{};\n"
        "    updateToken(sequence, true);\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        auto const attributionKvUpdateUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "            std::chrono::steady_clock::now() - attributionKvUpdateStarted).count();\n"
        "        auto const attributionAddTokenUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "            std::chrono::steady_clock::now() - attributionAddTokenStarted).count();\n"
        "        recordDataSystemRequestPhase(clientId, DataSystemRequestPhase::kAddToken, attributionAddTokenUs,\n"
        "            attributionLookupUs, attributionSequenceLookupUs, attributionKvUpdateUs);\n"
        "    }\n"
        "}\n")
    replace_supported(manager, (
        original_add_token,
        old_add_token,
        v2_add_token,
        legacy_v3_add_token,
    ), new_add_token)

    old_add_sequence = (
        "{\n"
        "    auto const clientId = llmRequest ? llmRequest->mClientId : std::optional<RequestIdType>{};\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    // Need to add the bubble after the sink tokens to use even block size\n")
    v2_add_sequence = (
        "{\n"
        "    auto const attributionEnabled = dataSystemRequestAttributionEnabled();\n"
        "    auto const clientId = attributionEnabled && llmRequest\n"
        "        ? llmRequest->mClientId\n"
        "        : std::optional<RequestIdType>{};\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    // Need to add the bubble after the sink tokens to use even block size\n")
    legacy_v3_add_sequence = (
        "{\n"
        "    auto const attributionAddSequenceStarted = std::chrono::steady_clock::now();\n"
        "    auto const attributionEnabled = dataSystemRequestAttributionEnabled();\n"
        "    auto const clientId = attributionEnabled && llmRequest\n"
        "        ? llmRequest->mClientId\n"
        "        : std::optional<RequestIdType>{};\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    // Need to add the bubble after the sink tokens to use even block size\n")
    new_add_sequence = (
        "{\n"
        "    auto const attributionEnabled = dataSystemRequestAttributionEnabled();\n"
        "    auto const attributionAddSequenceStarted = attributionEnabled\n"
        "        ? std::chrono::steady_clock::now()\n"
        "        : std::chrono::steady_clock::time_point{};\n"
        "    auto const clientId = attributionEnabled && llmRequest\n"
        "        ? llmRequest->mClientId\n"
        "        : std::optional<RequestIdType>{};\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    // Need to add the bubble after the sink tokens to use even block size\n")
    replace_supported(manager, (
        "{\n    // Need to add the bubble after the sink tokens to use even block size\n",
        old_add_sequence,
        v2_add_sequence,
        legacy_v3_add_sequence,
    ), new_add_sequence)

    old_add_sequence_store = (
        "    TLLM_CHECK(emplaceDone);\n"
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        mDataSystemClientIds[requestId] = clientId;\n"
        "    }\n"
        "    auto& sequence = seqIt->second;\n")
    new_add_sequence_store = (
        "    TLLM_CHECK(emplaceDone);\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        mDataSystemClientIds[requestId] = clientId;\n"
        "    }\n"
        "    auto& sequence = seqIt->second;\n")
    replace_supported(manager, (
        "    TLLM_CHECK(emplaceDone);\n    auto& sequence = seqIt->second;\n",
        old_add_sequence_store,
    ), new_add_sequence_store)

    add_sequence_timing = (
        "        llmRequest->updateMissedBlocksPerRequest(mBlockManager.getNumMissedBlocks() - numMissedBlocksPreRequest);\n"
        "    }\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        auto const attributionAddSequenceUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "            std::chrono::steady_clock::now() - attributionAddSequenceStarted).count();\n"
        "        recordDataSystemRequestPhase(\n"
        "            clientId, DataSystemRequestPhase::kAddSequence, attributionAddSequenceUs);\n"
        "    }\n"
        "}\n\nvoid KVCacheManager::storeContextBlocks")
    legacy_add_sequence_timing = (
        "        llmRequest->updateMissedBlocksPerRequest(mBlockManager.getNumMissedBlocks() - numMissedBlocksPreRequest);\n"
        "    }\n"
        "    auto const attributionAddSequenceUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "        std::chrono::steady_clock::now() - attributionAddSequenceStarted).count();\n"
        "    recordDataSystemRequestPhase(\n"
        "        clientId, DataSystemRequestPhase::kAddSequence, attributionAddSequenceUs);\n"
        "}\n\nvoid KVCacheManager::storeContextBlocks")
    replace_supported(manager, (
        "        llmRequest->updateMissedBlocksPerRequest(mBlockManager.getNumMissedBlocks() - numMissedBlocksPreRequest);\n"
        "    }\n"
        "}\n\nvoid KVCacheManager::storeContextBlocks",
        legacy_add_sequence_timing,
    ), add_sequence_timing)

    old_remove_sequence = (
        "void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)\n{\n"
        "    std::optional<RequestIdType> clientId = llmRequest ? llmRequest->mClientId : std::nullopt;\n"
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (!clientId && it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n")
    v2_remove_sequence = (
        "void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)\n{\n"
        "    auto const attributionEnabled = dataSystemRequestAttributionEnabled();\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        clientId = llmRequest ? llmRequest->mClientId : std::nullopt;\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (!clientId && it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n")
    legacy_v3_remove_sequence = (
        "void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)\n{\n"
        "    auto const attributionRemoveSequenceStarted = std::chrono::steady_clock::now();\n"
        "    auto const attributionEnabled = dataSystemRequestAttributionEnabled();\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        clientId = llmRequest ? llmRequest->mClientId : std::nullopt;\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (!clientId && it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n")
    new_remove_sequence = (
        "void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)\n{\n"
        "    auto const attributionEnabled = dataSystemRequestAttributionEnabled();\n"
        "    auto const attributionRemoveSequenceStarted = attributionEnabled\n"
        "        ? std::chrono::steady_clock::now()\n"
        "        : std::chrono::steady_clock::time_point{};\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        clientId = llmRequest ? llmRequest->mClientId : std::nullopt;\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (!clientId && it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n")
    replace_supported(manager, (
        "void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)\n{\n",
        old_remove_sequence,
        v2_remove_sequence,
        legacy_v3_remove_sequence,
    ), new_remove_sequence)

    old_remove_sequence_end = (
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        mDataSystemClientIds.erase(requestId);\n"
        "    }\n"
        "    if (clientId) finishDataSystemRequest(*clientId);\n"
        "    TLLM_LOG_TRACE(\"[%s]::%s stop\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n"
        "}\n\nvoid KVCacheManager::schedulingRemoveSequence")
    v2_remove_sequence_end = (
        "    if (attributionEnabled)\n"
        "    {\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        mDataSystemClientIds.erase(requestId);\n"
        "    }\n"
        "    if (clientId) finishDataSystemRequest(*clientId);\n"
        "    TLLM_LOG_TRACE(\"[%s]::%s stop\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n"
        "}\n\nvoid KVCacheManager::schedulingRemoveSequence")
    legacy_v3_remove_sequence_end = (
        "    if (attributionEnabled)\n"
        "    {\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        mDataSystemClientIds.erase(requestId);\n"
        "    }\n"
        "    auto const attributionRemoveSequenceUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "        std::chrono::steady_clock::now() - attributionRemoveSequenceStarted).count();\n"
        "    recordDataSystemRequestPhase(\n"
        "        clientId, DataSystemRequestPhase::kRemoveSequence, attributionRemoveSequenceUs);\n"
        "    if (clientId) finishDataSystemRequest(*clientId);\n"
        "    TLLM_LOG_TRACE(\"[%s]::%s stop\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n"
        "}\n\nvoid KVCacheManager::schedulingRemoveSequence")
    new_remove_sequence_end = (
        "    if (attributionEnabled)\n"
        "    {\n"
        "        std::scoped_lock lock(mDataSystemClientIdsMtx);\n"
        "        mDataSystemClientIds.erase(requestId);\n"
        "    }\n"
        "    if (attributionEnabled)\n"
        "    {\n"
        "        auto const attributionRemoveSequenceUs = std::chrono::duration_cast<std::chrono::microseconds>(\n"
        "            std::chrono::steady_clock::now() - attributionRemoveSequenceStarted).count();\n"
        "        recordDataSystemRequestPhase(\n"
        "            clientId, DataSystemRequestPhase::kRemoveSequence, attributionRemoveSequenceUs);\n"
        "    }\n"
        "    if (clientId) finishDataSystemRequest(*clientId);\n"
        "    TLLM_LOG_TRACE(\"[%s]::%s stop\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n"
        "}\n\nvoid KVCacheManager::schedulingRemoveSequence")
    replace_supported(manager, (
        "    TLLM_LOG_TRACE(\"[%s]::%s stop\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n}\n\nvoid KVCacheManager::schedulingRemoveSequence",
        old_remove_sequence_end,
        v2_remove_sequence_end,
        legacy_v3_remove_sequence_end,
    ), new_remove_sequence_end)

    transfer = source_dir / "kvCacheTransferManager.cpp"
    replace_once(transfer,
        '#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"\n',
        '#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"\n'
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n')
    text = transfer.read_text()
    if "#include <chrono>" not in text:
        replace_once(transfer, "#include <algorithm>\n", "#include <algorithm>\n#include <chrono>\n")

    malformed_debug = (
        "        TLLM_LOG_DEBUG(\"[TensorRT-LLM][Datasystem] mode = %d: pools.size() = %u, "
        "numTokensToCopy = %d.\",\n"
        "            pools.size(), numTokensToCopy);\n")
    corrected_debug = (
        "        TLLM_LOG_DEBUG(\"[TensorRT-LLM][Datasystem] mode = %d: pools.size() = %zu, "
        "numTokensToCopy = %d.\",\n"
        "            static_cast<int>(mode), pools.size(), numTokensToCopy);\n")
    transfer_text = transfer.read_text()
    if malformed_debug in transfer_text:
        replace_once(transfer, malformed_debug, corrected_debug)

    api_replacements = [
        (("datasystem::Status setRet = kvClient->Set(buffer);",), "setRet", "kSet", "setRet.IsError()",
         "attributionSet"),
        (("datasystem::Status getRet = kvClient1->Get("
          "std::to_string(BlockKeyHasher::hash(src->getBlockKey())), buffer, 0);",
          "datasystem::Status getRet = kvClient1->Get(key, buffer, 0);"),
         "getRet", "kGet", "getRet.IsError()", "attributionGet"),
        (("datasystem::Status setRet = kvClient->MSet(buffers);",), "setRet", "kSet", "setRet.IsError()",
         "attributionMSet"),
        (("datasystem::Status getRet = kvClient->Get(keys, buffers, 0);",
          "auto const getRet = kvClient->Get(keys, buffers, 0);"),
         "getRet", "kGet", "getRet.IsError()",
         "attributionMGet"),
    ]
    for calls, status, operation, failed, prefix in api_replacements:
        text = transfer.read_text()
        if f"auto const {prefix}Token =" in text:
            continue
        matching_calls = [call for call in calls if text.count(call) == 1]
        if len(matching_calls) != 1:
            raise RuntimeError(
                f"expected one supported DataSystem API form in {transfer}, "
                f"found {len(matching_calls)} for {prefix}: {calls!r}")
        call = matching_calls[0]
        wrapped = (
            f"auto const {prefix}Token = beginDataSystemOperation(DataSystemOperation::{operation});\n"
            f"            auto const {prefix}Started = std::chrono::steady_clock::now();\n"
            f"            {call}\n"
            f"            auto const {prefix}Us = std::chrono::duration_cast<std::chrono::microseconds>(\n"
            f"                std::chrono::steady_clock::now() - {prefix}Started).count();\n"
            f"            finishDataSystemOperation({prefix}Token, DataSystemOperation::{operation}, {prefix}Us, {failed});")
        replace_once(transfer, call, wrapped)

    patch_parallel_get(transfer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trtllm_dir", type=Path)
    args = parser.parse_args()
    root = args.trtllm_dir.resolve()
    patch_tree(root)
    print(f"TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION_PATCH_OK root={root}")


if __name__ == "__main__":
    main()
