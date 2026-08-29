import csv
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "scripts" / "summarize_pairec_rank_brpc_burst.py"
spec = importlib.util.spec_from_file_location("rank_burst_summary", SUMMARY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MARGINAL_SUMMARY = ROOT / "scripts" / "summarize_pairec_generation_kvc_rank_ab.py"
marginal_spec = importlib.util.spec_from_file_location(
    "generation_kvc_rank_ab_summary", MARGINAL_SUMMARY)
marginal_module = importlib.util.module_from_spec(marginal_spec)
marginal_spec.loader.exec_module(marginal_module)


class RankBRPCBurstSummaryTest(unittest.TestCase):
    def test_generation_kvc_rank_ab_wiring(self):
        benchmark = (
            ROOT / "scripts" / "benchmark_pairec_generation_kvc_rank_ab.sh"
        ).read_text()
        combined = (
            ROOT / "scripts" / "validate_pairec_brpc_wrapper_kvc_combined.sh"
        ).read_text()
        for token in (
            "WRAPPER_CONCURRENCY=1000",
            'GENERATION_BURST_POOL_SIZE=${GENERATION_BURST_POOL_SIZE:-10000}',
            "KVC_CONCURRENCY=32",
            "KVC_INPROCESS_PRESSURE=1",
            'RANK_ENDPOINT_OVERRIDE="$RANK_DIRECT_ENDPOINT"',
            "run_case rank_c1 1",
            "run_case rank_c1000 1000",
            "summarize_pairec_generation_kvc_rank_ab.py",
        ):
            self.assertIn(token, benchmark)
        for token in (
            'RANK_BURST_ENABLED=${RANK_BURST_ENABLED:-0}',
            "Wait for measured Rank burst completion",
            "PAIREC_COMBINED_RANK_BURST_DRAINED",
            'rank_complete["pressure_success"] == expected_rank_pressure',
            'rank_endpoint.get("rank_endpoint_source") == "override"',
            '"rank_front_brpc_ms"',
            '"rank_pressure_pipeline_overlap_ms"',
            'RANK_DEPLOYMENT=deepfm-rank-burst-wrapper',
            'does not contain rank-burst-wrapper',
            'PAIREC_COMBINED_RANK_WRAPPER_READY',
        ):
            self.assertIn(token, combined)

    def test_cross_node_log_collection_uses_bounded_lookback(self):
        combined = (
            ROOT / "scripts" / "validate_pairec_brpc_wrapper_kvc_combined.sh"
        ).read_text()
        full_chain = (
            ROOT / "scripts" / "deploy_and_validate_pairec_brpc_wrapper_full.sh"
        ).read_text()
        self.assertIn(
            'LOG_SINCE_LOOKBACK_SECONDS="${LOG_SINCE_LOOKBACK_SECONDS:-60}"',
            full_chain,
        )
        self.assertEqual(3, full_chain.count('--since-time="$LOG_SINCE_AT"'))
        self.assertNotIn('--since-time="$STARTED_AT"', full_chain)
        self.assertIn(
            'LOG_SINCE_LOOKBACK_SECONDS=${LOG_SINCE_LOOKBACK_SECONDS:-60}',
            combined,
        )
        self.assertIn(
            'LOG_SINCE_LOOKBACK_SECONDS="$LOG_SINCE_LOOKBACK_SECONDS"',
            combined,
        )
        self.assertEqual(7, combined.count('--since-time="$LOG_SINCE_AT"'))
        self.assertNotIn('--since-time="$STARTED_AT"', combined)

    def test_generation_kvc_rank_ab_summary_math(self):
        def case(concurrency, base):
            names = set(marginal_module.KEY_METRICS) | {
                "rank_pressure_success", "rank_pressure_errors"
            }
            metrics = {
                name: {"avg": base, "p99": base * 2, "max": base * 2}
                for name in names
            }
            metrics["rank_pressure_success"] = {
                "avg": float(concurrency - 1),
                "p99": float(concurrency - 1),
                "max": float(concurrency - 1),
            }
            metrics["rank_pressure_errors"] = {
                "avg": 0.0, "p99": 0.0, "max": 0.0,
            }
            return {
                "classification": "PAIREC_BRPC_WRAPPER_C1000_KVC_C32_OK",
                "wrapper_concurrency": 1000,
                "brpc_burst_pool_size": 10000,
                "kvc_concurrency": 32,
                "kvc_object_size_bytes": 3670016,
                "kvc_pressure_key_count": 4,
                "rank_burst_enabled": True,
                "rank_burst_concurrency": concurrency,
                "rank_business_payload_bytes": 102400,
                "rank_pressure_payload_bytes": 102400,
                "rank_endpoint": "192.168.100.11:18213",
                "rank_endpoint_source": "override",
                "samples": [{"request_id": "request-1"}],
                "metrics": metrics,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            control = root / "control.json"
            pressure = root / "pressure.json"
            control.write_text(json.dumps(case(1, 10.0)))
            pressure.write_text(json.dumps(case(1000, 25.0)))
            result = marginal_module.summarize(control, pressure)
        self.assertEqual(
            "PAIREC_GENERATION_C1000_KVC_C32_RANK_AB_OK",
            result["classification"],
        )
        self.assertEqual(15.0, result["comparison"]["client_e2e_ms"]["delta_avg"])
        self.assertEqual(30.0, result["comparison"]["rank_front_brpc_ms"]["delta_p99"])
        self.assertEqual(999, result["treatment"]["pressure_requests"])

    def test_worker_deploy_preflights_existing_engineering_model(self):
        script = (ROOT / "scripts" / "deploy_deepfm_rank_burst_worker1.sh").read_text()
        manifest = (ROOT / "k8s" / "deployment-deepfm-rank-brpc-worker1.yaml").read_text()
        wrapper_manifest = (
            ROOT / "k8s" / "deployment-deepfm-rank-burst-wrapper-worker1.yaml"
        ).read_text()
        ship = (ROOT / "scripts" / "ship_brpc_rank_burst_wrapper_binary_to_worker.sh").read_text()
        self.assertIn(
            "DEEPFM_MODEL_DIR:-/home/zcx/workspace/pairec4tigerllm/deepfm_out",
            script,
        )
        apply_position = script.index('kubectl apply -f "$OUTPUT_DIR/rank.yaml"')
        restart_position = script.index(
            'kubectl -n "$NAMESPACE" rollout restart "deployment/$RANK_DEPLOYMENT"'
        )
        status_position = script.index(
            'kubectl -n "$NAMESPACE" rollout status "deployment/$RANK_DEPLOYMENT"'
        )
        self.assertLess(apply_position, restart_position)
        self.assertLess(restart_position, status_position)
        for artifact in (
            "deepfm_best.pt",
            "feature_vocab.json",
            "user_profiles.json",
            "item_categories.json",
        ):
            self.assertLess(script.index(artifact), apply_position)
        self.assertLess(script.index("test -d '$BACKEND_REPO_DIR'"), apply_position)
        self.assertLess(script.index("test -s '$DEEPFM_MODEL_DIR/$artifact'"), apply_position)
        for binary in (
            "brpc_rank_burst_wrapper",
            "brpc_deepfm_rank_adapter",
            "brpc_pipeline_client",
        ):
            self.assertIn(binary, ship)
        self.assertIn("/home/zcx/bin/brpc_deepfm_rank_adapter", manifest)
        self.assertIn("/home/zcx/bin/brpc_pipeline_client", manifest)
        self.assertIn("/home/zcx/bin/brpc_pipeline_client", wrapper_manifest)
        self.assertIn(
            "mountPath: /opt/pairec-brpc/bin/brpc_pipeline_client",
            wrapper_manifest,
        )
        wrapper_source = (
            ROOT / "cpp" / "brpc_gateway" / "brpc_rank_burst_wrapper.cpp"
        ).read_text()
        self.assertIn("auto* trace = response->mutable_trace();", wrapper_source)
        self.assertIn(
            "trace->mutable_context()->CopyFrom(request->context());",
            wrapper_source,
        )
        self.assertIn('trace->set_component("brpc_rank_burst_wrapper");', wrapper_source)
        self.assertIn("trace->set_attribution_complete(true);", wrapper_source)
        self.assertIn("CPU placement diagnostics (non-blocking)", script)
        self.assertIn("cpu_isolation_valid=", script)

        self.assertNotIn(
            'assert not conflicts, "Rank CPU sets overlap',
            script,
        )

        benchmark = (ROOT / "scripts" / "benchmark_pairec_rank_brpc_burst.sh").read_text()
        self.assertIn("CPU isolation is not established", benchmark)
        self.assertNotIn(
            'assert json.load(open(sys.argv[1]))["valid"]',
            benchmark,
        )
        full_chain = (
            ROOT / "scripts" / "deploy_and_validate_pairec_brpc_wrapper_full.sh"
        ).read_text()
        self.assertIn("Verify preconnected Rank burst sessions", full_chain)
        self.assertIn("PAIREC_RANK_BRPC_BURST_PRECONNECTED_OK", full_chain)
        self.assertIn(
            "rebuild and import the PaiRec image containing the Rank burst coordinator",
            full_chain,
        )
        self.assertIn("warmup Rank burst completion timed out", full_chain)
        self.assertIn("all_enabled_bursts_drained=true", full_chain)
        self.assertIn('WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"', benchmark)
        self.assertIn("requires exactly one excluded warmup request", benchmark)
        self.assertIn(
            'RANK_DIRECT_ENDPOINT="${RANK_DIRECT_ENDPOINT:-192.168.100.11:18213}"',
            benchmark,
        )
        self.assertIn('RANK_ENDPOINT_OVERRIDE="$RANK_DIRECT_ENDPOINT"', benchmark)
        self.assertIn('RANK_ENDPOINT_OVERRIDE="${RANK_ENDPOINT_OVERRIDE:-}"', full_chain)
        self.assertIn('RANK_ENDPOINT_SOURCE="override"', full_chain)
        self.assertIn('RANK_ENDPOINT_SOURCE="service"', full_chain)
        self.assertIn('>"$OUTPUT_DIR/rank-endpoint.txt"', full_chain)

    def test_post_rank_host_network_rollout_does_not_overlap_ports(self):
        script = (ROOT / "scripts" / "deploy_post_rank_two_hop.sh").read_text()
        for name in ("deployment-post-rank-hop1.yaml", "deployment-post-rank-hop2.yaml"):
            manifest = (ROOT / "k8s" / name).read_text()
            self.assertIn("strategy:\n    type: Recreate", manifest)
            self.assertNotIn('limits:\n              cpu: "8"', manifest)
        self.assertIn("wait_pods_gone", script)
        self.assertIn("scale deployment/\"$deployment\" --replicas=0", script)
        self.assertIn("timed out deleting old hostNetwork pods", script)
        self.assertIn("failed to query old hostNetwork pods", script)
        self.assertIn("pairec_post_rank_binary_identity", script)
        self.assertNotIn("rollout restart deployment/post-rank-hop", script)

    def test_post_rank_business_lanes_wait_for_marker_not_pressure_drain(self):
        outer = (ROOT / "services" / "sort" / "deepfm_rank_sort.go").read_text()
        coordinator = (
            ROOT / "services" / "sort" / "brpc_rank_burst_coordinator.go"
        ).read_text()
        inner = (ROOT / "cpp" / "brpc_gateway" / "post_rank_hop_burst.cpp").read_text()
        self.assertIn("DedicatedBusinessLane: true", outer)
        self.assertIn("PressureStartQuorum:", outer)
        self.assertIn("MinimumLocalWindow:", outer)
        self.assertIn("businessLane = 1", coordinator)
        self.assertIn("pressure start marker timed out", coordinator)
        self.assertIn("LocalBusinessWindowMS", coordinator)
        self.assertIn("c.slot <- struct{}{}", coordinator)
        self.assertIn("std::deque<std::shared_ptr<Round>> rounds", inner)
        self.assertIn("business_mutex_", inner)
        self.assertIn("pressure_started_cv.wait_for", inner)
        self.assertIn("minimum_local_window_ms", inner)
        self.assertNotIn("round->ready == config_.concurrency - 1", inner)
        self.assertNotIn("previous post-rank hop2 burst is still active", inner)

    def test_post_rank_benchmark_has_latency_and_pressure_gates(self):
        benchmark = (
            ROOT / "scripts" / "benchmark_pairec_post_rank_two_hop_b_n1.sh"
        ).read_text()
        validator = (
            ROOT / "scripts" / "validate_pairec_brpc_wrapper_kvc_combined.sh"
        ).read_text()
        self.assertIn('MAX_E2E_MS="${MAX_E2E_MS:-550}"', benchmark)
        self.assertIn('MAX_POST_RANK_MS="${MAX_POST_RANK_MS:-120}"', benchmark)
        self.assertIn('MEASURED_REQUESTS="${MEASURED_REQUESTS:-3}"', benchmark)
        self.assertIn('"PAIREC_POST_RANK_TWO_HOP_B_N3_OK"', benchmark)
        self.assertIn('post_rank_hop1_local_business_window_ms', benchmark)
        self.assertIn('post_rank_hop2_local_business_window_ms', benchmark)
        self.assertIn('outer_complete["burst_valid"] is True', benchmark)
        self.assertIn("cpu_throttling=diagnostic", validator)
        self.assertNotIn('after["nr_throttled"]-before["nr_throttled"]==0', validator)

    def test_valid_pressure_case_and_tail_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with (root / "requests.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("index", "e2e_ms", "request_id", "response_end_epoch_ns"),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({"index": 1, "e2e_ms": 12.5, "request_id": "req-1",
                                 "response_end_epoch_ns": 360_000_000})
            pairec = [
                {"event": "pairec_rank_brpc_burst_start", "request_id": "req-1",
                 "concurrency": 4, "armed_workers": 4,
                 "business_payload_bytes": 102400, "pressure_payload_bytes": 102400},
                {"event": "pairec_rank_brpc_burst_business_complete", "request_id": "req-1",
                 "business_success": True, "trace_valid": True,
                 "business_payload_bytes": 102400, "business_client_wall_ms": 10,
                 "front_brpc_estimate_ms": 3, "rank_business_start_epoch_ns": 200_000_000,
                 "rank_business_end_epoch_ns": 300_000_000},
                {"event": "pairec_rank_brpc_burst_complete", "request_id": "req-1",
                 "pressure_requests": 3, "pressure_success": 3, "pressure_errors": 0,
                 "burst_valid": True, "pressure_overlap_business": 3,
                 "rank_pressure_first_start_epoch_ns": 190_000_000,
                 "rank_pressure_last_end_epoch_ns": 400_000_000,
                 "pressure_latency_p95_ms": 20, "max_active_workers": 4,
                 "start_skew_us": 5, "pressure_tail_after_business_ms": 100},
                {"event": "deepfm_rank_complete", "request_id": "req-1",
                 "candidate_count": 50, "reordered": True,
                 "model_version": "v1", "model_role": "engineering"},
                {"event": "source_quota_rerank_complete", "request_id": "req-1",
                 "status": "ok", "start_epoch_ns": 310_000_000,
                 "end_epoch_ns": 320_000_000},
                {"event": "pipeline_trace_complete", "request_id": "req-1",
                 "status": "ok", "valid": True, "start_epoch_ns": 0,
                 "end_epoch_ns": 350_000_000,
                 "spans": [{"name": "generative_recall", "status": "ok",
                            "start_offset_us": 10_000, "duration_us": 90_000}]},
            ]
            (root / "pairec-rank.log").write_text(
                "\n".join(json.dumps(event) for event in pairec) + "\n")
            (root / "inference.log").write_text(
                '1970-01-01T00:00:00.500000000Z '
                + json.dumps({"event": "trt_executor_request_complete", "request_id": "req-1"})
                + "\n")
            (root / "rank-wrapper.log").write_text(
                "[brpc-rank-burst-wrapper] method=Rank request_id=req-1 code=200 "
                "front_payload_bytes=102400 backend_payload_bytes=0 "
                "health_calls_during_rank=3 health_payload_bytes_during_rank=307200\n")

            result = module.summarize(root, 4, 102400, 102400)
            sample = result["samples"][0]
            self.assertEqual("PAIREC_RANK_BRPC_BURST_C4_OK", result["classification"])
            self.assertEqual(10.0, sample["rank_pressure_rerank_overlap_ms"])
            self.assertEqual(50.0, sample["rank_pressure_pipeline_overlap_ms"])
            self.assertEqual(40.0, sample["rank_pressure_tail_after_http_ms"])
            self.assertEqual(100.0, sample["inference_to_rank_gap_ms"])
            self.assertEqual(300.0, sample["cross_node_inference_log_delta_ms"])
            self.assertTrue(sample["inference_complete_before_rank"])


if __name__ == "__main__":
    unittest.main()
