import tempfile
import unittest
from pathlib import Path


class DataSystemAttributionPatcherTest(unittest.TestCase):
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
}
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
        self.assertIn("if (dataSystemRequestAttributionEnabled())", first)
        self.assertIn("if (attributionEnabled)", first)
        self.assertIn("std::scoped_lock lock(mDataSystemClientIdsMtx);", first)
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
