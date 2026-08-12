import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "scripts" / "extract_f19_historical_runtime_worker1.sh"
BENCHMARK = ROOT / "scripts" / "benchmark_f19_historical_runtime.sh"


class F19HistoricalRuntimeABTest(unittest.TestCase):
    def test_scripts_have_valid_shell_syntax(self):
        for script in (EXTRACT, BENCHMARK):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_extract_uses_containerd_without_mutating_image(self):
        text = EXTRACT.read_text()
        self.assertNotIn("images inspect", text)
        self.assertIn('images list -q', text)
        self.assertIn('IMAGE_TAR="${IMAGE_TAR:-/home/zcx/pairec-brpc-inference-k8s-arm64-trtllm-v1.tar}"', text)
        self.assertIn('images import "$IMAGE_TAR"', text)
        self.assertIn('images mount "$IMAGE" "$MOUNT_DIR"', text)
        self.assertIn('images unmount "$MOUNT_DIR"', text)
        self.assertIn("/opt/pairec-brpc/bin/brpc_inference_server", text)
        self.assertIn("/TensorRT-LLM/cpp/build/tensorrt_llm/libtensorrt_llm.so", text)
        self.assertIn("acba014e342030a57e1fba51fd691b1fbb7ffd3735488f03b28e08365b00dc43", text)
        self.assertIn("e0452812c00a56ae9a0b5817a5c0ca6b1a2e1b8e33dc63fe22c31e2e834010c4", text)

    def test_benchmark_pairs_artifacts_and_always_restores(self):
        text = BENCHMARK.read_text()
        self.assertIn("trap cleanup EXIT", text)
        self.assertIn("restore_current_runtime", text)
        self.assertIn('run_samples historical 0', text)
        self.assertIn('run_samples current 1', text)
        self.assertIn('container["command"] = [f"{pod_dir}/bin/brpc_inference_server"]', text)
        self.assertIn('{"name": "f19-historical-trt", "mountPath": trt_path', text)
        self.assertIn('if mount["name"] not in f19_names', text)
        self.assertIn("--type=merge", text)
        self.assertIn("warm_current_minus_historical", text)


if __name__ == "__main__":
    unittest.main()
