import importlib.util
import json
import pathlib
import tempfile
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ReverseBrpcBurstTest(unittest.TestCase):
    def test_coordinator_is_strict_preconnected_and_marker_gated(self):
        header = (ROOT / "cpp/brpc_gateway/reverse_burst.h").read_text()
        source = (ROOT / "cpp/brpc_gateway/reverse_burst.cpp").read_text()
        for token in (
            "int concurrency = 1000",
            "int payload_bytes = 102400",
            "int marker_timeout_ms = 1000",
            "int pressure_timeout_ms = 5000",
            "bool initially_armed = false",
        ):
            self.assertIn(token, header)
        for token in (
            'options.connection_type = "single"',
            'options.connection_group = config_.stage + "_lane_" + std::to_string(lane)',
            '"connection_groups\\\":" << config_.concurrency',
            'config.concurrency != 1000',
            'config.payload_bytes != 102400',
            '"marker" : "health"',
            "pairec_reverse_brpc_burst_marker_complete",
            "pairec_reverse_brpc_burst_complete",
            "previous reverse burst round is still active",
            "result->error = \"reverse burst marker timed out\"",
            "SinkEchoMatches(response.raw_json(), round->marker)",
            "reverse burst worker creation failed",
        ):
            self.assertIn(token, source)

    def test_wrappers_trigger_after_backend_and_fail_closed(self):
        generation = (ROOT / "cpp/brpc_gateway/brpc_burst_wrapper.cpp").read_text()
        rank = (ROOT / "cpp/brpc_gateway/brpc_rank_burst_wrapper.cpp").read_text()
        for source, stage, failure in (
            (generation, "generation_return", "generation reverse burst marker failed"),
            (rank, "rank_return", "rank reverse burst marker failed"),
        ):
            self.assertIn(stage, source)
            self.assertIn("reverse_burst_->Trigger(marker, &reverse_result)", source)
            self.assertIn(failure, source)
            self.assertIn("PAIREC_RETURN_CONTROL_V1", (
                ROOT / "cpp/brpc_gateway/reverse_burst.h").read_text())
        self.assertLess(
            generation.index("forwarder_->Recommend"),
            generation.index("reverse_burst_->Trigger"),
        )
        self.assertLess(rank.index("forwarder_->Rank"), rank.index("reverse_burst_->Trigger"))
        self.assertIn("rank_kvc_ms + reverse_result.wall_ms", rank)

    def test_two_master_sinks_are_direct_and_guaranteed(self):
        documents = list(yaml.safe_load_all((
            ROOT / "k8s/deployment-brpc-return-pressure-sinks-master.yaml"
        ).read_text()))
        self.assertEqual(2, len(documents))
        expected = {
            "generation-return-pressure-sink": (18301, "generation_return"),
            "rank-return-pressure-sink": (18302, "rank_return"),
        }
        for document in documents:
            name = document["metadata"]["name"]
            port, stage = expected[name]
            pod = document["spec"]["template"]["spec"]
            self.assertEqual("master", pod["nodeName"])
            self.assertTrue(pod["hostNetwork"])
            container = pod["containers"][0]
            self.assertIn(f"--listen_port={port}", container["args"])
            self.assertIn(f"--stage={stage}", container["args"])
            self.assertEqual(
                container["resources"]["requests"],
                container["resources"]["limits"],
            )
            self.assertEqual("8", container["resources"]["limits"]["cpu"])
            self.assertEqual("2Gi", container["resources"]["limits"]["memory"])

    def test_reverse_wrappers_use_master_25g_endpoint(self):
        generation = yaml.safe_load((
            ROOT / "k8s/deployment-brpc-burst-wrapper-188.yaml"
        ).read_text())
        rank = next(
            document for document in yaml.safe_load_all((
                ROOT / "k8s/deployment-deepfm-rank-burst-wrapper-worker1.yaml"
            ).read_text())
            if document.get("kind") == "Deployment"
        )
        generation_args = generation["spec"]["template"]["spec"]["containers"][0]["args"]
        rank_wrapper = next(
            item for item in rank["spec"]["template"]["spec"]["containers"]
            if item["name"] == "rank-burst-wrapper"
        )
        self.assertIn(
            "--reverse_burst_endpoint=192.168.100.12:18301", generation_args
        )
        self.assertIn(
            "--reverse_burst_endpoint=192.168.100.12:18302", rank_wrapper["args"]
        )

    def test_generation_control_client_is_shipped_and_host_mounted(self):
        ship = (
            ROOT / "scripts/ship_brpc_burst_wrapper_binary_to_worker.sh"
        ).read_text()
        deployment = yaml.safe_load((
            ROOT / "k8s/deployment-brpc-burst-wrapper-188.yaml"
        ).read_text())
        self.assertIn("BINARIES=(brpc_burst_wrapper brpc_recommend_client)", ship)
        mounts = deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        self.assertTrue(any(
            item["mountPath"] == "/opt/pairec-brpc/bin/brpc_recommend_client"
            for item in mounts
        ))

    def test_control_protocol_is_gated_before_remote_benchmark(self):
        build = (ROOT / "scripts/build_brpc_inference_image.sh").read_text()
        generation_ship = (
            ROOT / "scripts/ship_brpc_burst_wrapper_binary_to_worker.sh"
        ).read_text()
        rank_ship = (
            ROOT / "scripts/ship_brpc_rank_burst_wrapper_binary_to_worker.sh"
        ).read_text()
        benchmark = (
            ROOT / "scripts/benchmark_pairec_reverse_brpc_abba.sh"
        ).read_text()
        for source in (build, generation_ship, rank_ship, benchmark):
            self.assertIn("PAIREC_RETURN_CONTROL_V1", source)
        self.assertLess(
            benchmark.index("Preflight reverse BRPC control clients"),
            benchmark.index("Deploy master return pressure Sinks"),
        )
        self.assertIn("brpc_recommend_client brpc_pipeline_client", benchmark)
        generation_deploy = benchmark.index(
            "bash scripts/k8s_apply_brpc_burst_wrapper_188.sh"
        )
        previous_pool_drain = benchmark.index(
            "PAIREC_PREVIOUS_GENERATION_POOL_DRAINED"
        )
        self.assertLess(
            benchmark.index("Preflight reverse BRPC control clients"),
            previous_pool_drain,
        )
        self.assertLess(previous_pool_drain, generation_deploy)
        self.assertLess(
            generation_deploy,
            benchmark.index("Functional treatment smoke n1"),
        )

    def test_abba_uses_two_thousand_generation_sessions(self):
        benchmark = (
            ROOT / "scripts/benchmark_pairec_reverse_brpc_abba.sh"
        ).read_text()
        self.assertIn(
            'GENERATION_BURST_POOL_SIZE="${GENERATION_BURST_POOL_SIZE:-2000}"',
            benchmark,
        )
        self.assertIn(
            'BURST_POOL_SIZE="$GENERATION_BURST_POOL_SIZE"', benchmark
        )
        self.assertIn("GENERATION_BURST_POOL_SIZE >= 1000", benchmark)
        self.assertIn("BURST_ACTIVE_CONNECTIONS=1000", benchmark)
        self.assertIn(
            "scale deployment/pairec-brpc-observed-wrapper", benchmark
        )
        self.assertIn(
            "-l app=pairec-brpc-observed-wrapper --timeout=120s", benchmark
        )
        self.assertIn('SMOKE_ONLY="${SMOKE_ONLY:-0}"', benchmark)
        self.assertLess(
            benchmark.index("PAIREC_REVERSE_BRPC_FUNCTIONAL_SMOKE_OK"),
            benchmark.index("Interleaved reverse BRPC A/B"),
        )

    def test_abba_summary_splits_ten_and_ten(self):
        path = ROOT / "scripts/summarize_pairec_reverse_brpc_ab.py"
        spec = importlib.util.spec_from_file_location("reverse_ab", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pattern = [label for _ in range(5) for label in ("A", "B", "B", "A")]
        samples = []
        for index, label in enumerate(pattern):
            sample = {
                "request_id": f"request-{index}",
                "reverse_burst_treatment": float(label == "B"),
            }
            for name in module.KEY_METRICS:
                sample[name] = 20.0 if label == "B" else 10.0
            samples.append(sample)
        source = {
            "samples": samples,
            "reverse_burst_pattern": ",".join(pattern),
            "response_semantic_fingerprints": ["same"] * 20,
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = pathlib.Path(directory) / "input.json"
            input_path.write_text(json.dumps(source))
            result = module.summarize(input_path)
        self.assertEqual("PAIREC_REVERSE_BRPC_ABBA_OK", result["classification"])
        self.assertEqual(10, result["control_samples"])
        self.assertEqual(10, result["treatment_samples"])
        self.assertEqual(10.0, result["metrics"]["client_e2e_ms"]["delta_avg"])

    def test_round_drain_precedes_next_replay(self):
        contention = (ROOT / "scripts/benchmark_brpc_kvc_contention.sh").read_text()
        self.assertIn("wait_reverse_bursts", contention)
        self.assertIn("PAIREC_REVERSE_BRPC_DRAINED", contention)
        self.assertIn('complete["pressure_success"] == 999', contention)
        self.assertIn('sink["unique_lanes"] == 1000', contention)
        self.assertLess(
            contention.index('run_replay "$round_dir"', contention.index("for round in")),
            contention.index('wait_reverse_bursts "$round_dir"'),
        )

    def test_same_node_reverse_tail_overlap_metrics_are_reported(self):
        validator = (
            ROOT / "scripts/validate_pairec_brpc_wrapper_kvc_combined.sh"
        ).read_text()
        summary = (ROOT / "scripts/summarize_pairec_reverse_brpc_ab.py").read_text()
        for metric in (
            "generation_reverse_tail_rank_overlap_ms",
            "rank_reverse_tail_rerank_overlap_ms",
        ):
            self.assertIn(metric, validator)
            self.assertIn(metric, summary)
        self.assertIn('reverse_events["generation"]["wrapper_complete"]', validator)
        self.assertIn('reverse_events["rank"]["sink_complete"]', validator)
        self.assertIn("PAIREC_REVERSE_SINK_RUNTIME_OK", validator)


if __name__ == "__main__":
    unittest.main()
