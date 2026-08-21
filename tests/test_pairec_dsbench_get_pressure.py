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


if __name__ == "__main__":
    unittest.main()
