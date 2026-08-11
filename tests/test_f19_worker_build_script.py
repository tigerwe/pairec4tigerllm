import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_f19_attribution_runtime_worker1.sh"


class F19WorkerBuildScriptTest(unittest.TestCase):
    def test_build_uses_separate_trt_and_gateway_images(self):
        text = SCRIPT.read_text()
        self.assertIn(
            'TRT_BUILD_IMAGE="${TRT_BUILD_IMAGE:-zcx-pairec-image:v1.1}"',
            text)
        self.assertIn('GATEWAY_BUILD_IMAGE="${GATEWAY_BUILD_IMAGE:-}"', text)
        self.assertIn(
            "pairec-brpc-inference:k8s-arm64-trtllm-multisequence-kvc-ctx224-v1",
            text)
        self.assertIn("Stage 1/2: build TensorRT-LLM shared library", text)
        self.assertIn("Stage 2/2: build BRPC inference gateway", text)
        self.assertIn("--gpus all", text)
        self.assertIn("--entrypoint /bin/bash", text)
        self.assertIn('"$cuda_driver_dir:/host-driver:ro"', text)

    def test_gateway_image_is_preflighted_for_full_sdk(self):
        text = SCRIPT.read_text()
        self.assertIn("gateway_image_has_sdk()", text)
        self.assertIn("command -v protoc", text)
        self.assertIn("/usr/local/include/brpc/server.h", text)
        self.assertIn("/usr/local/include/google/protobuf/message.h", text)
        self.assertIn("no local arm64 gateway image contains protoc", text)

    def test_host_paths_use_known_container_mount_points(self):
        text = SCRIPT.read_text()
        self.assertIn('CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/mnt/pairec-src}"', text)
        self.assertIn('CONTAINER_TRTLLM_DIR="${CONTAINER_TRTLLM_DIR:-/TensorRT-LLM}"', text)
        self.assertIn('-v "$TRTLLM_DIR:$CONTAINER_TRTLLM_DIR"', text)
        self.assertNotIn('-v "$TRTLLM_DIR:$TRTLLM_DIR"', text)

    def test_default_parallelism_is_capped(self):
        text = SCRIPT.read_text()
        self.assertIn("if (( default_jobs > 32 )); then default_jobs=32; fi", text)
        self.assertIn('JOBS="${JOBS:-$default_jobs}"', text)

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
        self.assertIn('staging_dir="$(mktemp -d', text)
        self.assertLess(
            text.index('install -m 0755 "$gateway" /out/bin/brpc_inference_server'),
            text.index("== Publish verified F19 V2 runtime =="))
        self.assertGreater(
            text.rindex('"$RUNTIME_DIR/bin/brpc_inference_server"'),
            text.index("== Publish verified F19 V2 runtime =="))

    def test_v2_and_token_trace_are_hard_gates(self):
        text = SCRIPT.read_text()
        self.assertIn('APPLY_PATCH="${APPLY_PATCH:-1}"', text)
        self.assertIn(
            'apply_trtllm_datasystem_request_attribution_patch.sh', text)
        self.assertIn("ZERO_INTRUSION_DISABLED_V2", text)
        self.assertIn("datasystem_request_complete", text)
        self.assertIn('\\"version\\":2', text)
        self.assertIn("output_token_count", text)
        self.assertIn("runner_ms_per_output_token", text)


if __name__ == "__main__":
    unittest.main()
