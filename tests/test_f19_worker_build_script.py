import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_f19_attribution_runtime_worker1.sh"


class F19WorkerBuildScriptTest(unittest.TestCase):
    def test_build_uses_known_kvc_image_and_gpu_runtime(self):
        text = SCRIPT.read_text()
        self.assertIn(
            "pairec-brpc-inference:k8s-arm64-trtllm-multisequence-kvc-ctx224-v1",
            text)
        self.assertIn("--gpus all", text)
        self.assertIn("--entrypoint /bin/bash", text)
        self.assertIn("/host-driver/libcuda.so.1:ro", text)

    def test_build_restores_both_previous_link_requirements(self):
        text = SCRIPT.read_text()
        self.assertIn("libcudadevrt.a", text)
        self.assertIn("libcudart_static.a", text)
        self.assertIn('export LIBRARY_PATH="$cuda_static_dir:', text)

    def test_build_rebuilds_and_installs_both_overlay_artifacts(self):
        text = SCRIPT.read_text()
        self.assertIn(
            'cmake --build "$TRTLLM_DIR/cpp/build" --target tensorrt_llm', text)
        self.assertIn(
            'cmake --build "$gateway_build" --target brpc_inference_server', text)
        self.assertIn(
            'install -m 0755 "$gateway" /out/bin/brpc_inference_server', text)
        self.assertIn(
            'install -m 0755 "$trt_library" /out/lib/libtensorrt_llm.so', text)

    def test_v2_and_token_trace_are_hard_gates(self):
        text = SCRIPT.read_text()
        self.assertIn("ZERO_INTRUSION_DISABLED_V2", text)
        self.assertIn("datasystem_request_complete", text)
        self.assertIn("'\"version\":2'", text)
        self.assertIn("output_token_count", text)
        self.assertIn("runner_ms_per_output_token", text)


if __name__ == "__main__":
    unittest.main()
