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

class DataSystemRequestScope
{
public:
    explicit DataSystemRequestScope(std::optional<std::uint64_t> correlationId);
    ~DataSystemRequestScope();

    DataSystemRequestScope(DataSystemRequestScope const&) = delete;
    DataSystemRequestScope& operator=(DataSystemRequestScope const&) = delete;

private:
    std::optional<std::uint64_t> mPrevious;
};

[[nodiscard]] bool dataSystemRequestAttributionEnabled();
void registerDataSystemRequest(
    std::uint64_t correlationId, std::string const& requestId, std::uint64_t nativeLifecycleCount);
void finishDataSystemRequest(std::uint64_t correlationId);
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
            TLLM_LOG_INFO("{\"event\":\"datasystem_attribution_ready\",\"version\":1}");
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

DataSystemRequestScope::DataSystemRequestScope(std::optional<std::uint64_t> correlationId)
    : mPrevious(gCorrelationId)
{
    gCorrelationId = correlationId;
}

DataSystemRequestScope::~DataSystemRequestScope()
{
    gCorrelationId = mPrevious;
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
    replace_once(manager_h,
        "    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;\n",
        "    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;\n"
        "    // Exact Executor clientId used only for DataSystem request attribution.\n"
        "    std::unordered_map<LlmRequest::RequestIdType, std::optional<LlmRequest::RequestIdType>>\n"
        "        mDataSystemClientIds;\n")

    manager = source_dir / "kvCacheManager.cpp"
    replace_once(manager,
        '#include "tensorrt_llm/batch_manager/kvCacheManager.h"\n',
        '#include "tensorrt_llm/batch_manager/kvCacheManager.h"\n'
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n')
    replace_once(manager,
        "void KVCacheManager::addToken(RequestIdType requestId)\n{\n    auto& sequence = getSequence(requestId);\n",
        "void KVCacheManager::addToken(RequestIdType requestId)\n{\n"
        "    std::optional<RequestIdType> clientId;\n"
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    auto& sequence = getSequence(requestId);\n")
    replace_once(manager,
        "{\n    // Need to add the bubble after the sink tokens to use even block size\n",
        "{\n"
        "    auto const clientId = llmRequest ? llmRequest->mClientId : std::optional<RequestIdType>{};\n"
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    // Need to add the bubble after the sink tokens to use even block size\n")
    replace_once(manager,
        "    TLLM_CHECK(emplaceDone);\n    auto& sequence = seqIt->second;\n",
        "    TLLM_CHECK(emplaceDone);\n"
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        mDataSystemClientIds[requestId] = clientId;\n"
        "    }\n"
        "    auto& sequence = seqIt->second;\n")
    replace_once(manager,
        "void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)\n{\n",
        "void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)\n{\n"
        "    std::optional<RequestIdType> clientId = llmRequest ? llmRequest->mClientId : std::nullopt;\n"
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        auto const it = mDataSystemClientIds.find(requestId);\n"
        "        if (!clientId && it != mDataSystemClientIds.end()) clientId = it->second;\n"
        "    }\n"
        "    DataSystemRequestScope attributionScope(clientId);\n")
    replace_once(manager,
        "    TLLM_LOG_TRACE(\"[%s]::%s stop\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n}\n\nvoid KVCacheManager::schedulingRemoveSequence",
        "    {\n"
        "        std::scoped_lock lock(mSequencesMtx);\n"
        "        mDataSystemClientIds.erase(requestId);\n"
        "    }\n"
        "    if (clientId) finishDataSystemRequest(*clientId);\n"
        "    TLLM_LOG_TRACE(\"[%s]::%s stop\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n}\n\nvoid KVCacheManager::schedulingRemoveSequence")

    transfer = source_dir / "kvCacheTransferManager.cpp"
    replace_once(transfer,
        '#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"\n',
        '#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"\n'
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n')
    text = transfer.read_text()
    if "#include <chrono>" not in text:
        replace_once(transfer, "#include <algorithm>\n", "#include <algorithm>\n#include <chrono>\n")

    api_replacements = [
        (("datasystem::Status setRet = kvClient->Set(buffer);",), "setRet", "kSet", "setRet.IsError()",
         "attributionSet"),
        (("datasystem::Status getRet = kvClient1->Get("
          "std::to_string(BlockKeyHasher::hash(src->getBlockKey())), buffer, 0);",
          "datasystem::Status getRet = kvClient1->Get(key, buffer, 0);"),
         "getRet", "kGet", "getRet.IsError()", "attributionGet"),
        (("datasystem::Status setRet = kvClient->MSet(buffers);",), "setRet", "kSet", "setRet.IsError()",
         "attributionMSet"),
        (("datasystem::Status getRet = kvClient->Get(keys, buffers, 0);",), "getRet", "kGet", "getRet.IsError()",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trtllm_dir", type=Path)
    args = parser.parse_args()
    root = args.trtllm_dir.resolve()
    patch_tree(root)
    print(f"TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION_PATCH_OK root={root}")


if __name__ == "__main__":
    main()
