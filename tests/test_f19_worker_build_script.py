import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_f19_attribution_runtime_worker1.sh"
CMAKE = ROOT / "cpp" / "brpc_gateway" / "CMakeLists.txt"
GENERATED = ROOT / "cpp" / "brpc_gateway" / "generated"


class F19WorkerBuildScriptTest(unittest.TestCase):
    def test_build_uses_separate_trt_and_gateway_images(self):
        text = SCRIPT.read_text()
        self.assertIn('TRT_BUILD_IMAGE="${TRT_BUILD_IMAGE:-}"', text)
        self.assertIn('GATEWAY_BUILD_IMAGE="${GATEWAY_BUILD_IMAGE:-}"', text)
        self.assertIn(
            "zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1", text)
        self.assertIn(
            "pairec-brpc-inference:k8s-arm64-trtllm-multisequence-kvc-ctx224-v1",
            text)
        self.assertIn("Stage 1/2: build TensorRT-LLM shared library", text)
        self.assertIn("Stage 2/2: build BRPC inference gateway", text)
        self.assertIn("--gpus all", text)
        self.assertIn("--entrypoint /bin/bash", text)
        self.assertIn('"$cuda_driver_dir:/host-driver:ro"', text)

    def test_trt_image_is_preflighted_for_static_cuda_runtime(self):
        text = SCRIPT.read_text()
        self.assertIn("trt_image_has_toolchain()", text)
        self.assertIn('find -L "$cuda_root"', text)
        self.assertIn("TRT candidate rejected:", text)
        self.assertIn("-e LD_PRELOAD=", text)
        self.assertIn("libcudadevrt.a", text)
        self.assertIn("libcudart_static.a", text)
        self.assertIn(
            "no local arm64 TRT image contains compiler, libcudadevrt.a", text)
        sdk = text.index("zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1")
        base = text.index("zcx-pairec-image:v1.1", sdk)
        self.assertLess(sdk, base)

    def test_gateway_image_is_preflighted_for_full_sdk(self):
        text = SCRIPT.read_text()
        self.assertIn("gateway_image_has_sdk()", text)
        self.assertIn("/usr/local/include/brpc/server.h", text)
        self.assertIn("/usr/include/brpc/server.h", text)
        self.assertIn("/usr/local/include/google/protobuf/message.h", text)
        self.assertIn("/usr/include/google/protobuf/message.h", text)
        self.assertIn("no local arm64 gateway image contains protobuf", text)
        self.assertNotIn("command -v protoc", text)
        self.assertIn("-DPAIREC_USE_PREGENERATED_PROTO=ON", text)
        self.assertIn('-DBRPC_INCLUDE_DIR="$brpc_include"', text)
        self.assertIn('-DBRPC_LIBRARY="$brpc_library"', text)

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
        self.assertIn('\\"version\\":3', text)
        self.assertIn("PAIREC_TRT_EXECUTOR_PHASE_TIMING_V3", text)
        self.assertIn("phase_timing_complete", text)
        self.assertIn("trt_executor_request_complete", text)
        self.assertIn("output_token_count", text)
        self.assertIn("runner_ms_per_output_token", text)

    def test_pregenerated_proto_mode_is_explicit_and_complete(self):
        cmake = CMAKE.read_text()
        self.assertIn("option(PAIREC_USE_PREGENERATED_PROTO", cmake)
        self.assertIn("if(PAIREC_USE_PREGENERATED_PROTO)", cmake)
        self.assertIn("protoc 25.1 / Protobuf C++ 4.25.1", cmake)
        for name in (
                "recommend.pb.cc", "recommend.pb.h",
                "pipeline_service.pb.cc", "pipeline_service.pb.h"):
            self.assertTrue((GENERATED / name).is_file(), name)
        for name in ("recommend.pb.h", "pipeline_service.pb.h"):
            header = (GENERATED / name).read_text()
            self.assertIn("Protobuf C++ Version: 4.25.1", header)
            self.assertIn("generated_message_tctable_decl.h", header)
            self.assertNotIn("generated_message_table_driven.h", header)

    def test_pregenerated_sources_match_current_proto(self):
        protoc = shutil.which("protoc")
        if protoc is None:
            self.skipTest("protoc is unavailable for generated-source audit")
        version = subprocess.run(
            [protoc, "--version"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if version != "libprotoc 25.1":
            self.skipTest(f"generated sources require protoc 25.1, found {version}")
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([
                protoc,
                f"--proto_path={ROOT / 'proto'}",
                f"--cpp_out={directory}",
                str(ROOT / "proto" / "recommend.proto"),
                str(ROOT / "proto" / "pipeline_service.proto"),
            ], check=True)
            for name in (
                    "recommend.pb.cc", "recommend.pb.h",
                    "pipeline_service.pb.cc", "pipeline_service.pb.h"):
                self.assertEqual(
                    (GENERATED / name).read_bytes(),
                    (pathlib.Path(directory) / name).read_bytes(),
                    name)


if __name__ == "__main__":
    unittest.main()
