import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/recover_pairec_brpc_wrapper_kvc_environment.sh"


class RecoverPairecBrpcWrapperKvcEnvironmentTest(unittest.TestCase):
    def test_script_is_valid_and_preserves_recovery_guards(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text()
        self.assertIn("kill -CONT", text)
        self.assertIn("kill -TERM", text)
        self.assertNotIn("kill -KILL", text)
        self.assertNotIn("kill -9", text)
        self.assertIn('deploy_f14_kvc_burst_overlay.sh" restore', text)
        self.assertIn('rollout status "deployment/$DEPLOYMENT"', text)
        self.assertIn("kvc-burst-wrapper remains", text)
        self.assertIn("KVC_BURST environment remains", text)
        self.assertIn("PAIREC_BRPC_WRAPPER_KVC_ENVIRONMENT_RECOVERED", text)


if __name__ == "__main__":
    unittest.main()
