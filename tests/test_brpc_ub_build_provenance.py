import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class BrpcUbBuildProvenanceTest(unittest.TestCase):
    def test_audit_is_read_only_and_captures_reproduction_inputs(self):
        script = (ROOT / "scripts" / "audit_brpc_ub_echo_build_provenance.sh").read_text()

        self.assertIn("/home/zcx/workspace/brpc-827", script)
        self.assertIn("0947eeff3cdbdab635f34a3b3ff5f6d1", script)
        self.assertIn("/home/zcx/workspace/brpc", script)
        self.assertIn("02df77b0294ccdcf08a6a8a39050de3a", script)
        self.assertIn("tree-manifest.txt", script)
        self.assertIn("config-comparison.txt", script)
        self.assertIn("lockfile-diff.txt", script)
        self.assertIn("command-log-diff.txt", script)
        self.assertIn("MODULE.bazel.lock", script)
        self.assertIn("src/bthread/bthread.cpp", script)
        self.assertIn("src/brpc/channel.cpp", script)
        self.assertIn("src/brpc/ubsocket_initializer.cpp", script)
        self.assertIn("record_external_repositories", script)
        self.assertIn("command.log", script)
        self.assertIn("host-toolchain.txt", script)
        self.assertIn("SYSTEM_BAZELRC", script)
        self.assertIn("USER_BAZELRC", script)
        self.assertIn("bazel_sha256", script)
        self.assertIn("LD_LIBRARY_PATH_env", script)
        self.assertIn("Build ID:", script)
        self.assertIn("bind jetty success", script)
        self.assertIn("PASS_MARKERS_UNBOUND_TO_BINARY", script)
        self.assertIn("SHA256_BOUND_PASS", script)
        self.assertIn("INVALID_BTHREAD_KEY_CONFIRMED", script)
        self.assertIn("reproduce-success-build.txt", script)
        self.assertIn("verify_sha()", script)
        self.assertIn("LOCKFILE_OR_UNRECORDED_BUILD_GRAPH_DIFFERENCE", script)
        self.assertIn("preserved-success-artifacts", script)
        self.assertIn("echo_c++_server.$success_server_sha", script)
        self.assertIn("echo_c++_client.$success_client_sha", script)
        self.assertIn("--define=brpc_with_urma=true", script)
        self.assertIn("BRPC_UB_ECHO_PROVENANCE_AUDIT_OK", script)

        self.assertNotIn("sed -i", script)
        self.assertNotIn("git reset", script)
        self.assertNotIn("git checkout", script)
        self.assertNotIn("bthread_key_create(&ubsocket_trace", script)

    def test_generated_recipe_is_valid_bash_and_diffs_are_emitted(self):
        script = ROOT / "scripts" / "audit_brpc_ub_echo_build_provenance.sh"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = pathlib.Path(temporary_directory)
            success_root = temporary / "success-root"
            candidate_root = temporary / "candidate-root"
            success_output = temporary / "success-output"
            candidate_output = temporary / "candidate-output"
            output_dir = temporary / "audit"

            relevant_files = (
                ".bazelrc",
                "MODULE.bazel",
                "MODULE.bazel.lock",
                "local_deps_ext.bzl",
                "src/bthread/bthread.cpp",
                "src/bthread/key.cpp",
                "src/brpc/channel.cpp",
                "src/brpc/ubsocket_initializer.cpp",
                "example/BUILD.bazel",
                "example/echo_c++_client.cpp",
                "example/echo_c++_server.cpp",
            )
            for root in (success_root, candidate_root):
                for relative in relevant_files:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"fixture={relative}\n")
                (root / "bazel-bin" / "example").mkdir(parents=True)
                shutil.copy2("/bin/true", root / "bazel-bin" / "example" / "echo_c++_server")
                shutil.copy2("/bin/true", root / "bazel-bin" / "example" / "echo_c++_client")

            (candidate_root / "MODULE.bazel.lock").write_text("fixture=candidate-lock\n")
            for output_base, command in (
                (success_output, "bazel build historical\n"),
                (candidate_output, "bazel build candidate\n"),
            ):
                (output_base / "external").mkdir(parents=True)
                (output_base / "command.log").write_text(command)

            environment = os.environ.copy()
            environment.update(
                SUCCESS_ROOT=str(success_root),
                CANDIDATE_ROOT=str(candidate_root),
                SUCCESS_OUTPUT_BASE=str(success_output),
                CANDIDATE_OUTPUT_BASE=str(candidate_output),
                SUCCESS_SERVER_BIN=str(success_root / "bazel-bin/example/echo_c++_server"),
                SUCCESS_CLIENT_BIN=str(success_root / "bazel-bin/example/echo_c++_client"),
                CANDIDATE_SERVER_BIN=str(candidate_root / "bazel-bin/example/echo_c++_server"),
                CANDIDATE_CLIENT_BIN=str(candidate_root / "bazel-bin/example/echo_c++_client"),
                OUTPUT_DIR=str(output_dir),
            )
            result = subprocess.run(
                ["bash", str(script)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            recipe = output_dir / "reproduce-success-build.txt"
            subprocess.run(["bash", "-n", str(recipe)], check=True)
            self.assertIn("verify_sha", recipe.read_text())
            self.assertNotIn("$\\(1", recipe.read_text())
            self.assertIn("candidate-lock", (output_dir / "lockfile-diff.txt").read_text())
            self.assertIn("bazel build candidate", (output_dir / "command-log-diff.txt").read_text())
            self.assertIn("lockfile_comparison=DIFFERENT_OR_MISSING", result.stdout)


if __name__ == "__main__":
    unittest.main()
