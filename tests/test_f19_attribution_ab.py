import csv
import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SHELL_SCRIPT = ROOT / "scripts" / "benchmark_f19_attribution_ab.sh"
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_f19_attribution_ab.py"


class F19AttributionABTest(unittest.TestCase):
    def write_run(self, root, pair, mode, avg, p99, runner_avg, complete,
                  output_tokens=32):
        run_dir = root / f"pair-{pair}-{mode}" / "brpc"
        run_dir.mkdir(parents=True)
        summary = {
            "classification": "PAIREC_BRPC_PIPELINE_TRACE_OK",
            "valid_count": 2,
            "missing_count": 0,
            "invalid_count": 0,
            "datasystem_complete_count": complete,
            "client_e2e": {"avg_ms": avg, "p99_ms": p99},
            "service_phases": {
                "generative_recall.runner_generate_us": {
                    "avg_ms": runner_avg,
                    "p99_ms": runner_avg + 1,
                },
                "generative_recall.runner_per_output_token_us": {
                    "count": 2,
                    "avg_ms": runner_avg / output_tokens,
                    "p99_ms": (runner_avg + 1) / output_tokens,
                },
            },
            "service_counts": {
                "generative_recall.output_token_count": {
                    "count": 2,
                    "avg": output_tokens,
                    "p99": output_tokens,
                },
            },
            "datasystem": {
                "get_count": 2 if mode == "enabled" else 0,
                "set_count": 4 if mode == "enabled" else 0,
                "get": {"avg_ms": 1.0},
                "set": {"avg_ms": 2.0},
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        elapsed_seconds = 2 * avg / 1000.0
        (run_dir / "workload.json").write_text(
            json.dumps(
                {
                    "requests": 2,
                    "elapsed_seconds": elapsed_seconds,
                    "throughput_rps": 2 / elapsed_seconds,
                }
            )
        )
        with (run_dir / "requests.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("index", "e2e_ms", "request_id"), delimiter="\t"
            )
            writer.writeheader()
            writer.writerow({"index": 1, "e2e_ms": avg, "request_id": "a"})
            writer.writerow({"index": 2, "e2e_ms": avg, "request_id": "b"})

    def run_summary(self, enabled_delta):
        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        for pair in range(1, 4):
            self.write_run(root, pair, "disabled", 100.0, 110.0, 90.0, 0)
            self.write_run(
                root,
                pair,
                "enabled",
                100.0 + enabled_delta,
                110.0 + enabled_delta,
                90.0 + enabled_delta,
                2,
            )
        output = root / "summary.json"
        result = subprocess.run(
            [
                "python3",
                str(SUMMARY_SCRIPT),
                "--input-root",
                str(root),
                "--pairs",
                "3",
                "--expected-requests",
                "2",
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
        )
        return temporary, output, result

    def test_shell_contract_is_paired_and_reuses_images(self):
        text = SHELL_SCRIPT.read_text()
        self.assertIn("for mode in disabled enabled", text)
        self.assertIn('BUILD_IMAGES=0', text)
        self.assertIn('IMPORT_IMAGES=0', text)
        self.assertIn('FORCE_INFERENCE_RESTART=1', text)
        self.assertIn('REQUIRE_DATASYSTEM_ATTRIBUTION="$require_attribution"', text)
        self.assertIn('CLIENT_MAX_P99_MS=0', text)
        self.assertIn('NATIVE_DATASYSTEM_COMPLETIONS_OK phase=workload', text)

    def test_pair_median_passes_within_budget(self):
        temporary, output, result = self.run_summary(0.05)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(0, result.returncode, result.stderr)
        summary = json.loads(output.read_text())
        self.assertEqual("F19_ATTRIBUTION_AB_PASS", summary["classification"])
        self.assertTrue(all(summary["gates"].values()))

    def test_pair_median_fails_outside_average_budget(self):
        temporary, output, result = self.run_summary(0.2)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(1, result.returncode)
        summary = json.loads(output.read_text())
        self.assertEqual("F19_ATTRIBUTION_AB_FAIL", summary["classification"])
        self.assertFalse(summary["gates"]["avg_overhead"])

    def test_output_token_change_fails_semantic_gate(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        for pair in range(1, 4):
            self.write_run(root, pair, "disabled", 100.0, 110.0, 90.0, 0, 32)
            self.write_run(root, pair, "enabled", 100.0, 110.0, 90.0, 2, 31)
        output = root / "summary.json"
        result = subprocess.run([
            "python3", str(SUMMARY_SCRIPT), "--input-root", str(root),
            "--pairs", "3", "--expected-requests", "2",
            "--output", str(output),
        ], text=True, capture_output=True)
        self.assertEqual(1, result.returncode)
        summary = json.loads(output.read_text())
        self.assertFalse(summary["gates"]["output_token_semantics"])

    def test_historical_runner_regression_cannot_pass_pair_delta_gate(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        for pair in range(1, 4):
            self.write_run(root, pair, "disabled", 200.0, 210.0, 180.0, 0)
            self.write_run(root, pair, "enabled", 200.0, 210.0, 180.0, 2)
        output = root / "summary.json"
        result = subprocess.run([
            "python3", str(SUMMARY_SCRIPT), "--input-root", str(root),
            "--pairs", "3", "--expected-requests", "2",
            "--output", str(output),
        ], text=True, capture_output=True)
        self.assertEqual(1, result.returncode)
        summary = json.loads(output.read_text())
        self.assertFalse(summary["gates"]["absolute_runner_latency"])


if __name__ == "__main__":
    unittest.main()
