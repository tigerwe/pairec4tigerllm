import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PostRankHopStartupDiagnosticTest(unittest.TestCase):
    def test_collects_all_startup_boundaries_without_jq(self):
        script = (
            ROOT / "scripts" / "diagnose_post_rank_hop_startup_failure.sh"
        ).read_text()
        self.assertNotIn("jq ", script)
        self.assertIn("--previous --timestamps", script)
        self.assertIn("/proc/1/limits", script)
        self.assertIn("pids.current", script)
        self.assertIn("memory.events", script)
        self.assertIn("host-binary-sha256.txt", script)
        self.assertIn("host-binary-capabilities.txt", script)
        self.assertIn("ss -ltnp", script)
        self.assertIn("tcp-probes.json", script)

    def test_classifies_known_failure_boundaries(self):
        script = (
            ROOT / "scripts" / "diagnose_post_rank_hop_startup_failure.sh"
        ).read_text()
        for classification in (
            "POST_RANK_HOP_POD_NOT_CREATED",
            "POST_RANK_HOP_IMAGE_PULL_FAILURE",
            "POST_RANK_HOP_UNSCHEDULABLE",
            "POST_RANK_HOP_BINARY_ARGUMENT_CONTRACT_MISMATCH",
            "POST_RANK_HOP_STARTUP_OOM",
            "POST_RANK_HOP1_THREAD_RESOURCE_EXHAUSTION",
            "POST_RANK_HOP2_BUSINESS_ENDPOINT_UNREACHABLE",
            "POST_RANK_HOP1_PRECONNECT_TIMEOUT",
            "POST_RANK_HOP1_PRECONNECT_PARTIAL",
            "POST_RANK_HOST_PORT_CONFLICT",
            "POST_RANK_HOP1_PRECONNECT_INITIALIZATION_FAILURE",
        ):
            self.assertIn(classification, script)
        self.assertIn('root/"summary.json"', script)
        self.assertIn("PAIREC_POST_RANK_HOP_STARTUP_DIAGNOSTIC_COMPLETE", script)


if __name__ == "__main__":
    unittest.main()
