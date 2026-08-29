import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/recover_pairec_reverse_brpc_environment.sh"


class RecoverReverseBrpcEnvironmentTest(unittest.TestCase):
    def test_recovery_disables_both_wrappers_and_removes_sinks(self):
        source = SCRIPT.read_text()
        self.assertIn("set -euo pipefail", source)
        self.assertIn(
            "disable_reverse_endpoint brpc-burst-wrapper brpc-burst-wrapper",
            source,
        )
        self.assertIn(
            "disable_reverse_endpoint deepfm-rank-burst-wrapper rank-burst-wrapper",
            source,
        )
        self.assertIn('args[indexes[0]] = "--reverse_burst_endpoint="', source)
        self.assertIn("generation-return-pressure-sink", source)
        self.assertIn("rank-return-pressure-sink", source)
        self.assertIn("--ignore-not-found --wait=true", source)

    def test_recovery_waits_for_rollout_and_verifies_empty_endpoint(self):
        source = SCRIPT.read_text()
        self.assertIn('rollout status "deployment/$deployment"', source)
        self.assertIn('values != ["--reverse_burst_endpoint="]', source)
        self.assertIn("PAIREC_REVERSE_BRPC_ENVIRONMENT_RECOVERED", source)


if __name__ == "__main__":
    unittest.main()
