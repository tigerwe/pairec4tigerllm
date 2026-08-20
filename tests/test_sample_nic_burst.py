import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sample_nic_burst.py"
SPEC = importlib.util.spec_from_file_location("sample_nic_burst", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NicBurstSummaryTest(unittest.TestCase):
    def test_summary_reports_peak_and_percentile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "samples"
            path.write_text(
                "1000000000 0 0\n"
                "1020000000 25000000 50000000\n"
                "1040000000 50000000 75000000\n"
            )
            result = MODULE.summarize(path, 25e9)
        self.assertEqual(3, result["sample_count"])
        self.assertAlmostEqual(20.0, result["peak_tx_gbps"])
        self.assertAlmostEqual(80.0, result["peak_tx_link_pct"])
        self.assertAlmostEqual(10.0, result["peak_rx_gbps"])

    def test_contention_wires_sampler_around_replay(self):
        text = (ROOT / "scripts" / "benchmark_brpc_kvc_contention.sh").read_text()
        self.assertIn("sample_nic_burst.py' collect", text)
        self.assertIn("sample_nic_burst.py summarize", text)
        self.assertIn('KVC_NIC_BURST_INTERVAL_MS="${KVC_NIC_BURST_INTERVAL_MS:-20}"', text)

    def test_combined_summary_keeps_nic_samples(self):
        text = (ROOT / "scripts" / "validate_pairec_brpc_wrapper_kvc_combined.sh").read_text()
        self.assertIn('"nic_burst_samples": nic_burst_samples', text)


if __name__ == "__main__":
    unittest.main()
