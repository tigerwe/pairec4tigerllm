import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sample_tcp_queues.py"
SPEC = importlib.util.spec_from_file_location("sample_tcp_queues", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TcpQueueSummaryTest(unittest.TestCase):
    def test_summary_groups_connections_and_reports_queue_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "samples.jsonl"
            output = pathlib.Path(directory) / "summary.json"
            source.write_text(
                json.dumps(
                    {
                        "ts_ns": 1_000_000_000,
                        "role": "client",
                        "connections": [
                            {
                                "local_ip": "192.168.100.11",
                                "local_port": 50000,
                                "remote_ip": "192.168.100.12",
                                "remote_port": 18482,
                                "tx_queue_bytes": 10,
                                "rx_queue_bytes": 20,
                                "inode": 7,
                            }
                        ],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "ts_ns": 1_005_000_000,
                        "role": "client",
                        "connections": [
                            {
                                "local_ip": "192.168.100.11",
                                "local_port": 50000,
                                "remote_ip": "192.168.100.12",
                                "remote_port": 18482,
                                "tx_queue_bytes": 5,
                                "rx_queue_bytes": 40,
                                "inode": 7,
                            }
                        ],
                    }
                )
                + "\n"
            )
            result = MODULE.summarize(source)
            output.write_text(json.dumps(result))
        self.assertEqual(2, result["sample_count"])
        self.assertEqual(1, result["unique_connection_count"])
        self.assertEqual(1, result["queued_connection_count"])
        self.assertEqual(10, result["connections"][0]["peak_tx_queue_bytes"])
        self.assertEqual(40, result["connections"][0]["peak_rx_queue_bytes"])

    def test_contention_wires_tcp_sampler_around_replay(self):
        text = (ROOT / "scripts" / "benchmark_brpc_kvc_contention.sh").read_text()
        self.assertIn('KVC_TCP_QUEUE_SAMPLE="${KVC_TCP_QUEUE_SAMPLE:-0}"', text)
        self.assertIn("<scripts/sample_tcp_queues.py", text)
        self.assertIn("sample_tcp_queues.py summarize", text)

    def test_contention_wires_header_capture_and_owner_snapshots(self):
        text = (ROOT / "scripts" / "benchmark_brpc_kvc_contention.sh").read_text()
        self.assertIn('KVC_TCP_PACKET_SAMPLE="${KVC_TCP_PACKET_SAMPLE:-0}"', text)
        self.assertIn("tcp-owners-worker-before.txt", text)
        self.assertIn("tcp-owners-worker-after.txt", text)
        self.assertIn("tcpdump -i", text)
        self.assertIn("summarize_kvc_tcp_capture.py", text)
        self.assertIn(
            'KVC_TCP_PACKET_INTERFACE="${KVC_TCP_PACKET_INTERFACE:-$REMOTE_NETWORK_INTERFACE}"',
            text,
        )
        self.assertIn('ssh "$KVC_LOAD_HOST" "command -v tcpdump', text)


if __name__ == "__main__":
    unittest.main()
