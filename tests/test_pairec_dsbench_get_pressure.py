import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_pairec_dsbench_get_pressure.sh"


class PaiRecDsbenchGetPressureScriptTest(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_uses_gated_sustained_get_with_strict_pressure_evidence(self):
        text = SCRIPT.read_text()
        for token in (
            "MODE=kvc-get",
            "KVC_PRESSURE_ENGINE=dsbench",
            "KVC_DSBENCH_SUSTAINED=1",
            "KVC_GET_CLIENTS=\"$DSBENCH_CLIENTS\"",
            "KVC_GET_KEY_COUNT=\"$DSBENCH_KEY_COUNT\"",
            "REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1",
            "REQUIRE_BUSINESS_ONBOARD_GET=1",
            "RESET_INFERENCE_AFTER_PRIME=1",
            "RESET_INFERENCE_MODE=container-runtime",
            "KVC_NIC_BURST_SAMPLE=1",
            '"dsbench_get_max_inflight"',
            '"business_get_each_ms"',
            '"diagnostic_samples"',
        ):
            self.assertIn(token, text)

    def test_defaults_match_p64_comparison_shape(self):
        text = SCRIPT.read_text()
        self.assertIn('DSBENCH_CLIENTS="${DSBENCH_CLIENTS:-64}"', text)
        self.assertIn('DSBENCH_OBJECT_SIZE="${DSBENCH_OBJECT_SIZE:-3584KB}"', text)
        self.assertIn('PRIME_REQUESTS="${PRIME_REQUESTS:-195}"', text)

    def test_contention_resets_hbm_between_prime_and_pressure_release(self):
        contention = (ROOT / "scripts" / "benchmark_brpc_kvc_contention.sh").read_text()
        self.assertIn('RESET_INFERENCE_AFTER_PRIME="${RESET_INFERENCE_AFTER_PRIME:-0}"', contention)
        prime = contention.index('run_prime "$round_dir"')
        reset = contention.index('log "Reset inference HBM after DataSystem prime"')
        release = contention.rindex('release_kvc_load "$round_dir"')
        self.assertLess(prime, reset)
        self.assertLess(reset, release)


if __name__ == "__main__":
    unittest.main()
