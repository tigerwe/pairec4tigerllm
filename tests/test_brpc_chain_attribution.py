import unittest

from scripts.brpc_chain_attribution import attribute_brpc_chain


class BrpcChainAttributionTest(unittest.TestCase):
    def test_full_chain_includes_rank_coordination(self):
        summary = {
            "brpc_events": [{"latency_ms": "259"}],
            "pairec_generative_trace": {"protocol": "brpc", "rpc_ms": "287"},
            "pairec_json_events": [
                {
                    "event": "pipeline_trace_complete",
                    "spans": [
                        {"name": "vector_recall", "protocol": "brpc", "status": "ok",
                         "duration_us": 7531, "attributes": {"service_total_us": 6500}},
                        {"name": "deepfm_rank", "protocol": "brpc", "status": "ok",
                         "duration_us": 141122, "attributes": {"service_total_us": 7000}},
                    ],
                },
                {"event": "pairec_rank_brpc_burst_business_complete",
                 "front_brpc_estimate_ms": 35.410},
            ],
        }
        result = attribute_brpc_chain(summary)
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(28.0, result["generative_brpc_ms"])
        self.assertAlmostEqual(1.031, result["vector_brpc_ms"])
        self.assertAlmostEqual(134.122, result["rank_brpc_ms"])
        self.assertAlmostEqual(35.410, result["rank_business_brpc_ms"])
        self.assertAlmostEqual(98.712, result["rank_coordination_ms"])
        self.assertAlmostEqual(163.153, result["brpc_ms"])

    def test_burst_generation_event_takes_precedence(self):
        summary = {
            "brpc_events": [{"latency_ms": 200}],
            "pairec_generative_trace": {"protocol": "brpc", "rpc_ms": 250},
            "pairec_json_events": [
                {"event": "pairec_brpc_burst_business_complete",
                 "business_front_brpc_ms": 12.5},
                {"event": "pipeline_trace_complete", "spans": [
                    {"name": "vector_recall", "protocol": "brpc", "status": "ok",
                     "duration_us": 2000, "attributes": {"service_total_us": 1500}},
                    {"name": "deepfm_rank", "protocol": "brpc", "status": "ok",
                     "duration_us": 9000, "attributes": {"service_total_us": 7000}},
                ]},
            ],
        }
        result = attribute_brpc_chain(summary)
        self.assertEqual(12.5, result["generative_brpc_ms"])
        self.assertEqual(15.0, result["brpc_ms"])

    def test_missing_stages_are_explicit(self):
        result = attribute_brpc_chain({
            "brpc_events": [{"latency_ms": 100}],
            "pairec_generative_trace": {"protocol": "brpc", "rpc_ms": 110},
        })
        self.assertFalse(result["complete"])
        self.assertEqual("missing", result["sources"]["vector"])
        self.assertEqual("missing", result["sources"]["rank"])
        self.assertEqual(10.0, result["brpc_ms"])

    def test_legacy_generation_trace_without_protocol_is_supported(self):
        result = attribute_brpc_chain({
            "brpc_events": [{"latency_ms": 110}],
            "pairec_generative_trace": {"rpc_ms": 112},
        })
        self.assertEqual(2.0, result["generative_brpc_ms"])


if __name__ == "__main__":
    unittest.main()
