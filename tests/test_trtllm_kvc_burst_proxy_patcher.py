import tempfile
import unittest
from pathlib import Path


class KvcBurstProxyPatcherTest(unittest.TestCase):
    def test_patches_all_get_forms_idempotently(self):
        module = __import__("scripts.patch_trtllm_kvc_burst_proxy", fromlist=["patch_tree"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "cpp/include/tensorrt_llm/batch_manager"
            source = root / "cpp/tensorrt_llm/batch_manager"
            include.mkdir(parents=True)
            source.mkdir(parents=True)
            (include / "datasystemRequestTracker.h").write_text(
                "std::optional<std::string> currentDataSystemRequestId();\n")
            (source / "CMakeLists.txt").write_text("    datasystemRequestTracker.cpp\n")
            transfer = source / "kvCacheTransferManager.cpp"
            transfer.write_text(r'''#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"
#include <algorithm>
// TRTLLM_DATASYSTEM_PARALLEL_GET
void copy()
{
            auto const attributionGetToken = beginDataSystemOperation(DataSystemOperation::kGet);
            datasystem::Status getRet = kvClient1->Get(key, buffer, 0);
            finishDataSystemOperation(attributionGetToken, DataSystemOperation::kGet, attributionGetUs, getRet.IsError());
            auto const attributionMGetToken = beginDataSystemOperation(DataSystemOperation::kGet);
            auto const getRet = kvClient->Get(keys, buffers, 0);
            finishDataSystemOperation(attributionMGetToken, DataSystemOperation::kGet, attributionMGetUs, getRet.IsError());
                for (auto const& key : keys)
                {
                    auto const attributionParallelGetToken = 1;
                }
                activeParallelGetCallsAfter = gActiveParallelGetCalls.fetch_sub(1, std::memory_order_relaxed) - 1;
}
''')
            repo_root = Path(__file__).resolve().parents[1]
            module.patch_tree(root, repo_root)
            first = transfer.read_text()
            module.patch_tree(root, repo_root)
            second = transfer.read_text()

        self.assertEqual(first, second)
        self.assertIn("kvcBurstGetToken", first)
        self.assertIn("kvcBurstMGetToken", first)
        self.assertIn("kvcBurstParallelGetToken", first)
        self.assertIn("beginInProcessPressure", first)
        self.assertIn("kvcBurstGetPressure.finish()", first)
        self.assertIn("kvClient1->Get(pressureKey, pressureBuffer, 0)", first)
        self.assertIn("kvcBurstMGetPressure.finish()", first)
        self.assertIn("kvcBurstParallelGetPressure.finish()", first)
        self.assertIn("kvClient->Get(pressureKey, pressureBuffer, 0)", first)
        self.assertEqual(first.count("beginInProcessPressure"), 3)
        self.assertEqual(first.count("beginBusinessGet"), 3)
        self.assertEqual(first.count("finishBusinessGet"), 3)

    def test_requires_request_attribution_prerequisite(self):
        module = __import__("scripts.patch_trtllm_kvc_burst_proxy", fromlist=["patch_tree"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "cpp/include/tensorrt_llm/batch_manager"
            source = root / "cpp/tensorrt_llm/batch_manager"
            include.mkdir(parents=True)
            source.mkdir(parents=True)
            (include / "datasystemRequestTracker.h").write_text("#pragma once\n")
            (source / "CMakeLists.txt").write_text("    datasystemRequestTracker.cpp\n")
            (source / "kvCacheTransferManager.cpp").write_text("source\n")
            with self.assertRaisesRegex(RuntimeError, "attribution patch"):
                module.patch_tree(root, Path(__file__).resolve().parents[1])

    def test_upgrades_v3_managed_files_to_v5(self):
        module = __import__("scripts.patch_trtllm_kvc_burst_proxy", fromlist=["write_managed"])
        with tempfile.TemporaryDirectory() as directory:
            managed = Path(directory) / "managed.cpp"
            managed.write_text("/* PAIREC_KVC_BURST_PROXY_V3 */\nold\n")
            module.write_managed(managed, "new\n")
            result = managed.read_text()

        self.assertIn("PAIREC_KVC_BURST_PROXY_V5", result)
        self.assertNotIn("PAIREC_KVC_BURST_PROXY_V3", result)
        self.assertTrue(result.endswith("new\n"))


if __name__ == "__main__":
    unittest.main()
