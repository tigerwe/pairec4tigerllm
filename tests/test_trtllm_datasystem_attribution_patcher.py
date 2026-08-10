import tempfile
import unittest
from pathlib import Path


class DataSystemAttributionPatcherTest(unittest.TestCase):
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
