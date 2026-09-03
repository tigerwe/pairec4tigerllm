import pathlib
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
        self.assertIn("INVALID_BTHREAD_KEY_CONFIRMED", script)
        self.assertIn("reproduce-success-build.txt", script)
        self.assertIn("preserved-success-artifacts", script)
        self.assertIn("echo_c++_server.$success_server_sha", script)
        self.assertIn("echo_c++_client.$success_client_sha", script)
        self.assertIn("--define=brpc_with_urma=true", script)
        self.assertIn("BRPC_UB_ECHO_PROVENANCE_AUDIT_OK", script)

        self.assertNotIn("sed -i", script)
        self.assertNotIn("git reset", script)
        self.assertNotIn("git checkout", script)
        self.assertNotIn("bthread_key_create(&ubsocket_trace", script)


if __name__ == "__main__":
    unittest.main()
