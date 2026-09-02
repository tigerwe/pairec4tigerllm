import os
import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "cpp" / "brpc_ub_probe"


class BrpcUbRecommendProbeTest(unittest.TestCase):
    def test_transport_is_explicitly_enabled_on_both_ends(self):
        server = (PROBE_DIR / "minimal_recommend_server.cpp").read_text()
        client = (PROBE_DIR / "minimal_recommend_client.cpp").read_text()
        self.assertIn("options.use_ub = FLAGS_ubsocket_use_ub", server)
        self.assertIn("options.use_ub = FLAGS_ubsocket_use_ub", client)
        self.assertIn("response_attachment().append(payload)", server)
        self.assertIn("echoedPayload != payload", client)
        self.assertIn("options.connect_timeout_ms = FLAGS_probe_connect_timeout_ms", client)

        for script_name in ("run_brpc_ub_recommend_server.sh", "run_brpc_ub_recommend_matrix.sh"):
            script = (ROOT / "scripts" / script_name).read_text()
            self.assertIn("--ubsocket_backup_link_enable=false", script)
            self.assertIn("--ubsocket_degrade_enable=false", script)

    def test_payload_matrix_contains_boundary_and_large_values(self):
        client = (PROBE_DIR / "minimal_recommend_client.cpp").read_text()
        self.assertIn("0,1,4096,4097,65536,1048576,3670016", client)
        self.assertIn("Sha256Hex(payload)", client)
        self.assertIn("actualMetadata != expectedMetadata", client)

    def test_bazel_package_reuses_brpc_target(self):
        build = (PROBE_DIR / "BUILD.bazel").read_text()
        self.assertIn('"//:brpc"', build)
        self.assertIn('"-std=c++17"', build)
        self.assertNotIn("datasystem", build.lower())
        self.assertNotIn("trt", build.lower())

    def test_build_script_stages_complete_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update({"BRPC_ROOT": temp_dir, "SKIP_VERSION_CHECK": "1", "STAGE_ONLY": "1"})
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "build_brpc_ub_recommend_probe.sh")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            package = pathlib.Path(temp_dir) / "pairec_ub_probe"
            self.assertIn("BRPC_UB_RECOMMEND_STAGE_OK", result.stdout)
            self.assertEqual(
                {path.name for path in package.iterdir()},
                {
                    "BUILD.bazel",
                    "minimal_recommend_server.cpp",
                    "minimal_recommend_client.cpp",
                    "payload_integrity.h",
                    "recommend.proto",
                },
            )

    @unittest.skipUnless(shutil.which("g++"), "g++ is required")
    def test_payload_helper_compiles_and_hashes(self):
        source = textwrap.dedent(
            """
            #include "payload_integrity.h"
            #include <iostream>

            int main()
            {
                const std::string payload = pairec::brpc_ub_probe::BuildDeterministicPayload(4097, 7);
                const std::string digest = pairec::brpc_ub_probe::Sha256Hex(payload);
                std::cout << payload.size() << " " << digest.size() << "\\n";
                return payload.size() == 4097 && digest.size() == 64 ? 0 : 1;
            }
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = pathlib.Path(temp_dir) / "payload_smoke.cpp"
            binary_path = pathlib.Path(temp_dir) / "payload_smoke"
            source_path.write_text(source)
            compile_result = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-I",
                    str(PROBE_DIR),
                    str(source_path),
                    "-lcrypto",
                    "-o",
                    str(binary_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            result = subprocess.run([str(binary_path)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "4097 64")


if __name__ == "__main__":
    unittest.main()
