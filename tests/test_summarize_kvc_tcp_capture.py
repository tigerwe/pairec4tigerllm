import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_kvc_tcp_capture.py"
SPEC = importlib.util.spec_from_file_location("summarize_kvc_tcp_capture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KvcTcpCaptureSummaryTest(unittest.TestCase):
    def test_aligns_payload_by_connection_and_business_window(self):
        burst = {
            "request_id": "request-1",
            "business_get_start_epoch_ns": 2_000_000_000,
            "business_get_end_epoch_ns": 2_010_000_000,
            "pressure_start_epoch_ns": [1_995_000_000],
            "pressure_end_epoch_ns": [2_020_000_000],
        }
        with tempfile.TemporaryDirectory() as directory:
            capture = pathlib.Path(directory) / "capture.log"
            capture.write_text(
                "1.999000 IP 192.168.100.12.18482 > 192.168.100.11.50000: "
                "Flags [.], seq 1:101, ack 1, win 1, length 100\n"
                "2.005000 IP 192.168.100.12.18482 > 192.168.100.11.50000: "
                "Flags [.], seq 101:301, ack 1, win 1, length 200\n"
                "2.006000 IP 192.168.100.11.50000 > 192.168.100.12.18482: "
                "Flags [P.], seq 1:51, ack 301, win 1, length 50\n"
                "2.015000 IP 192.168.100.12.18482 > 192.168.100.11.50000: "
                "Flags [.], seq 301:701, ack 51, win 1, length 400\n"
            )
            result = MODULE.summarize(
                capture, burst, "192.168.100.12:18482", bucket_ms=1
            )
        self.assertEqual(4, result["parsed_packet_count"])
        self.assertEqual(1, result["unique_connection_count"])
        row = result["connections"][0]
        self.assertEqual(100, row["server_to_client_before_business_bytes"])
        self.assertEqual(200, row["server_to_client_business_window_bytes"])
        self.assertEqual(400, row["server_to_client_after_business_bytes"])
        self.assertEqual(50, row["client_to_server_business_window_bytes"])
        self.assertEqual(700, row["server_to_client_pressure_window_bytes"])

    def test_parses_tcpdump_fraction_as_epoch_nanoseconds(self):
        packet = MODULE.parse_packet(
            "1787293766.123456 IP 192.168.100.11.50000 > "
            "192.168.100.12.18482: Flags [.], length 1448"
        )
        self.assertEqual(1_787_293_766_123_456_000, packet["ts_ns"])


if __name__ == "__main__":
    unittest.main()
