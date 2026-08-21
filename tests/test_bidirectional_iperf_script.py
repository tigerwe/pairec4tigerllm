import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_pairec_bidirectional_iperf.sh"


class BidirectionalIperfScriptTest(unittest.TestCase):
    def test_script_runs_two_bound_directions_and_exact_replay(self):
        text = SCRIPT.read_text()
        self.assertIn('-B "$MASTER_DATA_IP"', text)
        self.assertIn('"$WORKER_SERVER_CPUS" "$WORKER_DATA_IP"', text)
        self.assertIn('-B "$bind_ip"', text)
        self.assertIn('master-to-worker.log', text)
        self.assertIn('worker-to-master.log', text)
        self.assertIn('REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1', text)
        self.assertIn('"get_count", 0', text)

    def test_script_primes_and_cleans_only_owned_processes(self):
        text = SCRIPT.read_text()
        self.assertIn('benchmark_go_brpc_probe_kvc_latency.sh', text)
        self.assertIn('REMOTE_SERVER_PID_FILE', text)
        self.assertIn('REMOTE_CLIENT_PID_FILE', text)
        self.assertNotIn('pkill', text)
        self.assertIn('trap cleanup EXIT', text)


if __name__ == "__main__":
    unittest.main()
