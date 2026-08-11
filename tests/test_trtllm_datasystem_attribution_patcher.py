import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class DataSystemAttributionPatcherTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "C++ compiler is required")
    def test_tracker_phase_timing_runtime_smoke(self):
        module = __import__(
            "scripts.patch_trtllm_datasystem_request_attribution",
            fromlist=["TRACKER_HEADER", "TRACKER_SOURCE"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "include"
            tracker_include = include / "tensorrt_llm/batch_manager"
            logger_include = include / "tensorrt_llm/common"
            tracker_include.mkdir(parents=True)
            logger_include.mkdir(parents=True)
            (tracker_include / "datasystemRequestTracker.h").write_text(
                module.TRACKER_HEADER)
            (logger_include / "logger.h").write_text(
                "#pragma once\n#include <cstdio>\n"
                "namespace fake_logger { template <typename... Args> "
                "void log(char const* format, Args... args) { "
                "std::printf(format, args...); std::printf(\"\\n\"); } }\n"
                "#define TLLM_LOG_INFO(...) fake_logger::log(__VA_ARGS__)\n"
                "#define TLLM_LOG_WARNING(...) fake_logger::log(__VA_ARGS__)\n")
            source = root / "datasystemRequestTracker.cpp"
            source.write_text(module.TRACKER_SOURCE)
            main = root / "main.cpp"
            main.write_text(r'''
#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"
#include <chrono>
#include <cstdlib>
#include <thread>
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
int main()
{
    setenv("TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION", "1", 1);
    registerDataSystemRequest(7, "request-7", 1);
    markDataSystemRequestExecutorEnqueued(7);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    recordDataSystemRequestPhase(7, DataSystemRequestPhase::kAddSequence, 20);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    recordDataSystemRequestPhase(7, DataSystemRequestPhase::kAddToken, 30, 2, 3, 20);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    recordDataSystemRequestPhase(7, DataSystemRequestPhase::kRemoveSequence, 10);
    finishDataSystemRequest(7);
}
''')
            binary = root / "tracker-smoke"
            subprocess.run([
                "c++", "-std=c++17", "-pthread", f"-I{include}",
                str(source), str(main), "-o", str(binary),
            ], check=True, capture_output=True, text=True)
            result = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True)

        events = [json.loads(line) for line in result.stdout.splitlines()
                  if '"event":"datasystem_request_complete"' in line]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["request_id"], "request-7")
        self.assertEqual(event["add_sequence_count"], 1)
        self.assertEqual(event["add_token_count"], 1)
        self.assertEqual(event["remove_sequence_count"], 1)
        self.assertEqual(event["native_lifecycle_count"], 1)
        self.assertEqual(event["phase_unknown_count"], 0)
        self.assertTrue(event["phase_timing_complete"])
        self.assertGreater(event["native_lifecycle_us"], 0)

    def test_existing_v1_tree_is_upgraded_to_v2_idempotently(self):
        module = __import__(
            "scripts.patch_trtllm_datasystem_request_attribution",
            fromlist=["patch_tree"])
        old_add_token = """void KVCacheManager::addToken(RequestIdType requestId)
{
    std::optional<RequestIdType> clientId;
    {
        std::scoped_lock lock(mSequencesMtx);
        auto const it = mDataSystemClientIds.find(requestId);
        if (it != mDataSystemClientIds.end()) clientId = it->second;
    }
    DataSystemRequestScope attributionScope(clientId);
    auto& sequence = getSequence(requestId);
    updateToken(sequence, true);
}
"""
        old_add_sequence = """void KVCacheManager::addSequence()
{
    auto const clientId = llmRequest ? llmRequest->mClientId : std::optional<RequestIdType>{};
    DataSystemRequestScope attributionScope(clientId);
    // Need to add the bubble after the sink tokens to use even block size
    TLLM_CHECK(emplaceDone);
    {
        std::scoped_lock lock(mSequencesMtx);
        mDataSystemClientIds[requestId] = clientId;
    }
    auto& sequence = seqIt->second;
    if (llmRequest)
    {
        llmRequest->updateMissedBlocksPerRequest(mBlockManager.getNumMissedBlocks() - numMissedBlocksPreRequest);
    }
}

void KVCacheManager::storeContextBlocks() {}
"""
        old_remove_sequence = """void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)
{
    std::optional<RequestIdType> clientId = llmRequest ? llmRequest->mClientId : std::nullopt;
    {
        std::scoped_lock lock(mSequencesMtx);
        auto const it = mDataSystemClientIds.find(requestId);
        if (!clientId && it != mDataSystemClientIds.end()) clientId = it->second;
    }
    DataSystemRequestScope attributionScope(clientId);
    {
        std::scoped_lock lock(mSequencesMtx);
        mDataSystemClientIds.erase(requestId);
    }
    if (clientId) finishDataSystemRequest(*clientId);
    TLLM_LOG_TRACE("[%s]::%s stop", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
}

void KVCacheManager::schedulingRemoveSequence() {}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "cpp/include/tensorrt_llm/batch_manager"
            source = root / "cpp/tensorrt_llm/batch_manager"
            include.mkdir(parents=True)
            source.mkdir(parents=True)
            (source / "CMakeLists.txt").write_text(
                "    dataTransceiverImpl.cpp\n    datasystemRequestTracker.cpp\n")
            (include / "kvCacheManager.h").write_text(
                "    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;\n"
                "    // Exact Executor clientId used only for DataSystem request attribution.\n"
                "    std::unordered_map<LlmRequest::RequestIdType, std::optional<LlmRequest::RequestIdType>>\n"
                "        mDataSystemClientIds;\n")
            manager = source / "kvCacheManager.cpp"
            manager.write_text(
                '#include "tensorrt_llm/batch_manager/kvCacheManager.h"\n'
                '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n'
                + old_add_token + old_add_sequence + old_remove_sequence)
            (source / "kvCacheTransferManager.cpp").write_text(
                '#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"\n'
                '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n'
                '#include <algorithm>\n#include <chrono>\n'
                'auto const attributionSetToken = 0;\n'
                'auto const attributionGetToken = 0;\n'
                'auto const attributionMSetToken = 0;\n'
                'auto const attributionMGetToken = 0;\n')

            module.patch_tree(root)
            first = manager.read_text()
            module.patch_tree(root)
            second = manager.read_text()

        self.assertEqual(first, second)
        self.assertIn(
            "PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2",
            first)
        self.assertIn("auto const attributionEnabled = dataSystemRequestAttributionEnabled();", first)
        self.assertIn("if (attributionEnabled)", first)
        self.assertIn("if (attributionEnabled)", first)
        self.assertIn("std::scoped_lock lock(mDataSystemClientIdsMtx);", first)
        self.assertIn("PAIREC_TRT_EXECUTOR_PHASE_TIMING_V3", first)
        self.assertIn("recordDataSystemRequestPhase", first)
        self.assertIn("attributionLookupUs", first)
        add_token = first.split("void KVCacheManager::addToken", 1)[1].split(
            "void KVCacheManager::addSequence", 1)[0]
        self.assertNotIn("mSequencesMtx", add_token)

    def test_disabled_path_is_guarded_before_sequence_map_lock(self):
        module = __import__(
            "scripts.patch_trtllm_datasystem_request_attribution",
            fromlist=["TRACKER_HEADER", "TRACKER_SOURCE"])
        patcher = Path(module.__file__).read_text()
        self.assertIn("bool mActive{false};", module.TRACKER_HEADER)
        self.assertIn("mActive = dataSystemRequestAttributionEnabled();", module.TRACKER_SOURCE)
        self.assertIn(
            "PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2",
            patcher)
        gate = patcher.index('"    if (dataSystemRequestAttributionEnabled())\\n"')
        lock = patcher.index(
            '"        std::scoped_lock lock(mDataSystemClientIdsMtx);\\n"', gate)
        self.assertLess(gate, lock)
        self.assertIn("legacy_v3_add_token", patcher)
        self.assertIn("legacy_v3_add_sequence", patcher)
        self.assertIn("legacy_v3_remove_sequence", patcher)

        gateway = Path("cpp/brpc_gateway/brpc_inference_server.cpp").read_text()
        self.assertIn(
            "datasystem_lifecycle ? &executor_timing : nullptr", gateway)
        self.assertIn("if (executor_timing_ptr)", gateway)
        self.assertIn("const auto request_setup_started = phase_timing", gateway)

    def test_replace_supported_upgrades_nested_old_anchor(self):
        module = __import__(
            "scripts.patch_trtllm_datasystem_request_attribution",
            fromlist=["replace_supported"])
        old = "function()\n{\n    old body\n"
        original = "function()\n{\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.cpp"
            path.write_text(old)
            module.replace_supported(path, (original, old), "new body\n")
            self.assertEqual("new body\n", path.read_text())

    def test_historical_parallel_get_shape_is_patched_idempotently(self):
        module = __import__(
            "scripts.patch_trtllm_datasystem_request_attribution",
            fromlist=["patch_parallel_get"])
        source = """// TRTLLM_DATASYSTEM_PARALLEL_GET
                for (auto const& key : keys)
                {
                    futures.emplace_back(dataSystemGetExecutor().submit([kvClient, key, monotonicMs]() mutable {
                        DataSystemGetResult result;
                        auto const started = monotonicMs();
                        result.getMs = monotonicMs() - started;
                        return result;
                    }));
                }
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kvCacheTransferManager.cpp"
            path.write_text(source)
            module.patch_parallel_get(path)
            first = path.read_text()
            module.patch_parallel_get(path)
            second = path.read_text()

        self.assertEqual(first, second)
        self.assertIn("attributionParallelGetToken", first)
        self.assertIn("attributionParallelGetStarted", first)
        self.assertIn("finishDataSystemOperation", first)
        self.assertIn("!result.ok", first)


if __name__ == "__main__":
    unittest.main()
