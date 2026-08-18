import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WRAPPER_CPP = ROOT / "cpp" / "kvc_burst" / "kvc_burst_wrapper.cpp"
SHARED_H = ROOT / "cpp" / "kvc_burst" / "kvc_burst_shared.h"
DEPLOY = ROOT / "scripts" / "deploy_f14_kvc_burst_overlay.sh"
CONTENTION = ROOT / "scripts" / "benchmark_brpc_kvc_contention.sh"
COMBINED = ROOT / "scripts" / "validate_pairec_brpc_wrapper_kvc_combined.sh"
COLLECTOR = ROOT / "scripts" / "collect_datasystem_worker_metrics.sh"
METRICS_PY = ROOT / "scripts" / "datasystem_worker_metrics.py"
DATASYSTEM_STUB = ROOT / "tests" / "stubs"
BUSINESS_PROBE_CPP = ROOT / "cpp" / "kvc_burst" / "kvc_burst_business_probe.cpp"

spec = importlib.util.spec_from_file_location("datasystem_worker_metrics", METRICS_PY)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

RAW_SNAPSHOT = """@cpu_stat
usage_usec 200000000
nr_periods 1000
nr_throttled 5
@cpuacct
usage
@loadavg
1.50 1.20 0.90 2/400 12345
@ctxt
voluntary_ctxt_switches:\t100000
nonvoluntary_ctxt_switches:\t5000
@procstat
1 (datasystem_worker) S 0 1 1 0 -1 4194304 100 0 0 0 4000 2000 0 0 20 0 1 0 50 1000000 100 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 6 0 0 0 0 0
@netdev
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000      10    0    0    0     0          0         0     1000      10    0    0    0     0       0          0
 enp41s0f1: 5000000  5000    0    0    0     0          0         0  8000000   6000    0    0    0     0       0          0
@softirq
                    CPU0       CPU1
          NET_RX:         10         20
          NET_TX:          5          7
@psi_cpu
some avg10=1.00 avg60=0.50 avg300=0.10 total=1000000
@psi_memory
some avg10=0.20 avg60=0.10 avg300=0.05 total=500000
full avg10=0.00 avg60=0.00 avg300=0.00 total=0
@memstat
pgfault 9000
pgmajfault 12
@memcurrent
1073741824
"""

RAW_SNAPSHOT_AFTER = RAW_SNAPSHOT.replace(
    "usage_usec 200000000", "usage_usec 240000000"
).replace(
    "voluntary_ctxt_switches:\t100000", "voluntary_ctxt_switches:\t160000"
).replace(
    "nonvoluntary_ctxt_switches:\t5000", "nonvoluntary_ctxt_switches:\t8000"
).replace(
    "4000 2000 0 0 20", "4400 2200 0 0 20"
).replace(
    "enp41s0f1: 5000000", "enp41s0f1: 25000000"
).replace(
    "0  8000000   6000", "0 36000000   6000"
).replace(
    "NET_RX:         10         20", "NET_RX:         110        220"
).replace(
    "NET_TX:          5          7", "NET_TX:          55         77"
).replace(
    "some avg10=1.00 avg60=0.50 avg300=0.10 total=1000000",
    "some avg10=2.00 avg60=0.60 avg300=0.10 total=1600000",
).replace(
    "pgfault 9000", "pgfault 19000"
).replace(
    "pgmajfault 12", "pgmajfault 14"
)


class DataSystemWorkerMetricsTest(unittest.TestCase):
    def test_parse_snapshot(self):
        snap = metrics.parse_snapshot(RAW_SNAPSHOT, "ds-pod", "master", 1_000_000_000)
        self.assertEqual("ds-pod", snap["pod"])
        self.assertEqual("master", snap["node"])
        self.assertEqual(200000000, snap["cpu_stat"]["usage_usec"])
        self.assertEqual(1.5, snap["loadavg"]["load1"])
        self.assertEqual(100000, snap["ctxt_switches"]["voluntary_ctxt_switches"])
        self.assertEqual(4000, snap["proc_stat"]["utime_ticks"])
        self.assertEqual(2000, snap["proc_stat"]["stime_ticks"])
        self.assertEqual(5000000, snap["netdev"]["enp41s0f1"]["rx_bytes"])
        self.assertEqual(8000000, snap["netdev"]["enp41s0f1"]["tx_bytes"])
        self.assertEqual(30, snap["softirq"]["net_rx"])
        self.assertEqual(12, snap["softirq"]["net_tx"])
        self.assertEqual(1000000.0, snap["psi_cpu"]["some"]["total"])
        self.assertEqual(12, snap["memory_stat"]["pgmajfault"])
        self.assertEqual(1073741824, snap["memory_current_bytes"])

    def test_delta(self):
        before = metrics.parse_snapshot(RAW_SNAPSHOT, "ds-pod", "master", 1_000_000_000)
        after = metrics.parse_snapshot(RAW_SNAPSHOT_AFTER, "ds-pod", "master", 11_000_000_000)
        delta = metrics.compute_delta(before, after)
        self.assertEqual(10.0, delta["elapsed_s"])
        self.assertFalse(delta["pod_changed"])
        # 40 s of cgroup CPU (usage_usec delta) over 10 s -> 4.0 cores.
        self.assertAlmostEqual(4.0, delta["cpu"]["avg_cores"])
        # (4400+2200-4000-2000) ticks / 100 tps / 10 s = 0.6 cores.
        self.assertAlmostEqual(0.6, delta["cpu"]["worker_process_avg_cores"])
        # 60000 switches over 10 s -> 6000/s.
        self.assertEqual(6000.0, delta["ctxt_switches"]["voluntary_ctxt_switches_per_s"])
        self.assertEqual(2000000.0, delta["netdev"]["enp41s0f1"]["rx_Bps"])
        self.assertEqual(2800000.0, delta["netdev"]["enp41s0f1"]["tx_Bps"])
        self.assertEqual(30.0, delta["softirq"]["net_rx_per_s"])
        self.assertEqual(12.0, delta["softirq"]["net_tx_per_s"])
        self.assertEqual(60000.0, delta["psi"]["psi_cpu"]["some_stall_us_per_s"])
        self.assertEqual(2.0, delta["psi"]["psi_cpu"]["after_avg10"])
        self.assertEqual(0.2, delta["memory"]["pgmajfault_per_s"])
        self.assertEqual(1073741824, delta["memory"]["current_bytes_after"])

    def test_module_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            raw = tmp_path / "snap.raw"
            raw.write_text(RAW_SNAPSHOT)
            out = tmp_path / "snap.json"
            subprocess.run(
                [sys.executable, str(METRICS_PY), "parse-snapshot", "--raw", str(raw),
                 "--pod", "p", "--node", "n", "--ts-ns", "1000", "--out", str(out)],
                check=True, capture_output=True, text=True)
            parsed = json.loads(out.read_text())
            self.assertEqual("p", parsed["pod"])


class SustainedPressureStructureTest(unittest.TestCase):
    def test_apply_script_requires_v3_managed_proxy(self):
        script = (ROOT / "scripts/apply_trtllm_kvc_burst_proxy_patch.sh").read_text()
        self.assertIn("grep -q PAIREC_KVC_BURST_PROXY_V3", script)
        self.assertNotIn("grep -q PAIREC_KVC_BURST_PROXY_V2", script)

    def test_shared_header_helpers(self):
        text = SHARED_H.read_text()
        self.assertIn("enum class SustainedStop", text)
        self.assertIn("SustainedStopReason", text)
        self.assertIn("BusinessSubmitRank", text)
        self.assertIn("WaitForPressureStarted", text)
        self.assertIn("RequiredPressureFirst", text)
        self.assertIn("kPressureNotEstablished", text)
        self.assertIn("constexpr uint32_t kVersion = 3", text)
        self.assertNotIn("uint32_t reserved", text)

    def test_wrapper_sustained_flow(self):
        text = WRAPPER_CPP.read_text()
        for token in (
            '"sustained_pressure"',
            '"sustained_max_duration_ms"',
            '"sustained_max_loops"',
            "SustainedStopReason(*control, generation, loopStarted",
            "sustainedStats->loops += 1;",
            '\\"business_submit_rank\\"',
            '\\"pressure_inflight_at_business_start\\"',
            "FetchAdd(&control->pressure_started_lanes, 1U)",
            '\\"pressure_start_offsets_us\\"',
            '\\"pressure_end_offsets_us\\"',
            '\\"sustained_loop_gets\\"',
            '\\"sustained_window_ms\\"',
        ):
            self.assertIn(token, text)
        # Sustained errors must be logged at most once per lane per generation.
        self.assertIn("errorLogged", text)

    def test_kvc_binaries_compile_with_sdk_shape_stub(self):
        for source in (WRAPPER_CPP, BUSINESS_PROBE_CPP):
            subprocess.run(
                ["g++", "-std=c++17", "-pthread", "-fsyntax-only",
                 "-I", str(DATASYSTEM_STUB), "-I", str(ROOT / "cpp" / "kvc_burst"),
                 str(source)],
                check=True, capture_output=True, text=True)

    def test_deploy_script_sustained_wiring(self):
        text = DEPLOY.read_text()
        for token in (
            "SUSTAINED_PRESSURE=${SUSTAINED_PRESSURE:-0}",
            "SUSTAINED_MAX_DURATION_MS=${SUSTAINED_MAX_DURATION_MS:-1000}",
            "SUSTAINED_MAX_LOOPS=${SUSTAINED_MAX_LOOPS:-100}",
            "PRESSURE_LEAD_US=${PRESSURE_LEAD_US:-1000}",
            'die "SUSTAINED_PRESSURE must be 0 or 1"',
            "--sustained_pressure=",
            "--sustained_max_duration_ms=",
            "--sustained_max_loops=",
            "--pressure_lead_us=",
            "expected version=3",
            '\"sustained_pressure\":true',
        ):
            self.assertIn(token, text)

    def test_deploy_patch_contains_sustained_args(self):
        text = DEPLOY.read_text()
        blocks = [b for b in re.findall(r"<<'PY'\n(.*?)\nPY", text, re.DOTALL)
                  if "sustained_pressure" in b]
        self.assertEqual(1, len(blocks))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            deployment = {
                "spec": {"template": {"spec": {"containers": [{
                    "name": "brpc-inference",
                    "image": "img:v1",
                    "env": [
                        {"name": "LD_PRELOAD", "value": "a/block_ds_consumer.so b/stub_gpu.so "
                            "c/libabseil_dll.so d/libnvidia-ml.so"},
                        {"name": "DATASYSTEM_HOST", "value": "x"},
                    ],
                }]}}}
            }
            deployment_path = tmp_path / "deployment.json"
            patch_path = tmp_path / "patch.json"
            deployment_path.write_text(json.dumps(deployment))
            argv = [
                sys.executable, "-c", blocks[0],
                str(deployment_path), str(patch_path), "brpc-inference", "kvc-burst-wrapper",
                "/host", "/pod", "100", "8", "1835008", "5", "1000", "ds", "18482", "1", "0", "1", "0",
                "1", "500", "42",
            ]
            subprocess.run(argv, check=True, capture_output=True, text=True)
            patch = json.loads(patch_path.read_text())
        containers = patch["spec"]["template"]["spec"]["containers"]
        sidecar = next(c for c in containers if c["name"] == "kvc-burst-wrapper")
        self.assertIn("--sustained_pressure=true", sidecar["args"])
        self.assertIn("--sustained_max_duration_ms=500", sidecar["args"])
        self.assertIn("--sustained_max_loops=42", sidecar["args"])
        self.assertIn("--pressure_lead_us=1000", sidecar["args"])

    def test_contention_ds_worker_hook(self):
        text = CONTENTION.read_text()
        self.assertIn('DS_WORKER_METRICS="${DS_WORKER_METRICS:-1}"', text)
        self.assertIn('DS_WORKER_POD_SELECTOR="${DS_WORKER_POD_SELECTOR:-app=datasystem-25g-master}"', text)
        self.assertIn("capture_datasystem_worker_metrics \"$round_dir\" before", text)
        self.assertIn("capture_datasystem_worker_metrics \"$round_dir\" after", text)
        self.assertIn("collect_datasystem_worker_metrics.sh snapshot", text)
        self.assertIn("datasystem-worker-metrics.error", text)

    def test_contention_replay_retry_wiring(self):
        text = CONTENTION.read_text()
        for token in (
            'REPLAY_MAX_ATTEMPTS="${REPLAY_MAX_ATTEMPTS:-2}"',
            'REPLAY_RETRY_PRIME_REQUESTS="${REPLAY_RETRY_PRIME_REQUESTS:-20}"',
            'REPLAY_RETRY_CHURN_REQUESTS="${REPLAY_RETRY_CHURN_REQUESTS:-$REPLAY_RETRY_PRIME_REQUESTS}"',
            'die "REPLAY_MAX_ATTEMPTS must be a positive integer"',
            "replay_zero_business_get \"$round_dir\"",
            "zero business onboard Gets; prepare target KV eviction and retry",
            '"${round_dir}/replay-attempt-${replay_attempt}"',
            '"${round_dir}/replay.attempts"',
            'prepare_replay_onboard_retry "$round_dir" "$replay_attempt"',
            'set_kvc_burst_arm "$round_dir" refresh',
            'USER_ID="$REPLAY_USER_ID"',
            'PRIME_UIDS="$churn_uids"',
            'PRIME_REQUESTS="$REPLAY_RETRY_CHURN_REQUESTS"',
            'set_kvc_burst_arm "$round_dir" verify-and-arm',
            'pressure_key_control=refresh,target-prime,churn,verify-and-arm',
        ):
            self.assertIn(token, text)

        target = text.index('OUT_DIR="$target_dir"')
        churn = text.index('run_prime_once "$churn_dir"')
        verify = text.index('set_kvc_burst_arm "$round_dir" verify-and-arm')
        self.assertLess(target, churn)
        self.assertLess(churn, verify)

    def test_replay_zero_business_get_detector(self):
        text = CONTENTION.read_text()
        blocks = re.findall(r"python3 - \"\$trt_log\" <<'PY'\n(.*?)\nPY", text, re.DOTALL)
        self.assertEqual(1, len(blocks))
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "brpc_trtllm.log"
            # Real TRT logs use compact JSON without spaces after colons.
            def compact(event):
                return json.dumps(event, separators=(",", ":"))

            event = {"event": "datasystem_request_complete", "request_id": "r1",
                     "get_count": 0, "set_count": 3}
            log.write_text('noise ' + compact(event) + '\n')
            result = subprocess.run([sys.executable, "-c", blocks[0], str(log)],
                                    capture_output=True)
            self.assertEqual(0, result.returncode)
            event["get_count"] = 2
            log.write_text(compact(event) + '\n')
            result = subprocess.run([sys.executable, "-c", blocks[0], str(log)],
                                    capture_output=True)
            self.assertEqual(1, result.returncode)
            log.write_text("no events here\n")
            result = subprocess.run([sys.executable, "-c", blocks[0], str(log)],
                                    capture_output=True)
            self.assertEqual(1, result.returncode)

    def test_combined_script_sustained_wiring(self):
        text = COMBINED.read_text()
        for token in (
            "KVC_SUSTAINED_PRESSURE=${KVC_SUSTAINED_PRESSURE:-0}",
            'die "KVC_SUSTAINED_PRESSURE must be 0 or 1"',
            'SUSTAINED_PRESSURE="$KVC_SUSTAINED_PRESSURE"',
            'PRESSURE_LEAD_US="$KVC_PRESSURE_LEAD_US"',
            '"kvc_business_submit_rank"',
            '"kvc_pressure_inflight_at_business_start"',
            '"kvc_sustained_loop_gets"',
            '"datasystem_worker_samples"',
            'kvc.get("sustained_enabled", False) is kvc_sustained_pressure',
            'kvc["pressure_inflight_at_business_start"] >= required_pressure_first',
        ):
            self.assertIn(token, text)

    def test_shell_syntax(self):
        for script in (DEPLOY, CONTENTION, COMBINED, COLLECTOR):
            subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)

    def test_embedded_python_compiles(self):
        for script in (DEPLOY, CONTENTION, COMBINED):
            text = script.read_text()
            for block in re.findall(r"<<'PY'\n(.*?)\nPY", text, re.DOTALL):
                compile(block, str(script), "exec")
            for block in re.findall(r"<<'PY' \| tee \"\$SUMMARY_TXT\"\n(.*?)\nPY", text, re.DOTALL):
                compile(block, str(script), "exec")


if __name__ == "__main__":
    unittest.main()
