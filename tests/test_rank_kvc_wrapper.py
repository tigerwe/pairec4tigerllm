import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RankKvcWrapperTest(unittest.TestCase):
    def test_rank_wrapper_uses_one_shared_real_kvc_client(self):
        source = (ROOT / "cpp/brpc_gateway/brpc_rank_burst_wrapper.cpp").read_text()
        proxy = (ROOT / "cpp/kvc_burst/kvc_operation_proxy.cpp").read_text()
        for token in (
            "std::unique_ptr<datasystem::KVClient> client_",
            "client_->Get(key, buffer, timeout_ms)",
            "buffer->GetSize()",
            "buffer->ImmutableData()",
            "rank_kvc_business_timeout_ms = 1000",
            "rank_kvc_pressure_timeout_ms = 2000",
            "rank KVC business Get failed",
            "rank_kvc_business_get_complete",
            "rank_kvc_preflight_complete",
            "registerInProcessPressureClient",
            "response->mutable_trace()->set_total_us",
            'startup_stage = "rank_kvc_init"',
            "Rank burst wrapper startup exception: stage=",
            "value.size() < prefix.size()",
        ):
            self.assertIn(token, source)
        self.assertIn("expectedBusinessGets() == 1U ? lanes", proxy)
        self.assertIn("expectedBusinessGets() == 2U && started == 1U", proxy)
        self.assertIn("expectedGets == 2U", proxy)

    def test_rank_kvc_sidecar_is_independent_and_exact_shape(self):
        path = ROOT / "k8s/deployment-deepfm-rank-burst-wrapper-worker1.yaml"
        docs = list(yaml.safe_load_all(path.read_text()))
        containers = docs[0]["spec"]["template"]["spec"]["containers"]
        wrapper = next(item for item in containers if item["name"] == "rank-burst-wrapper")
        sidecar = next(item for item in containers if item["name"] == "rank-kvc-burst-wrapper")
        env = {item["name"]: item["value"] for item in wrapper["env"]}
        self.assertEqual("/run/pairec-rank-kvc-burst/control", env["KVC_BURST_CONTROL_PATH"])
        self.assertEqual("1", env["KVC_BURST_EXPECTED_BUSINESS_GETS"])
        self.assertEqual("8388608", env["KVC_BURST_EXPECTED_OBJECT_BYTES"])
        args = sidecar["args"]
        for expected in (
            "--concurrency=__RANK_KVC_CONCURRENCY__",
            "--pressure_key_count=__RANK_KVC_PRESSURE_KEY_COUNT__",
            "--object_size=8388608",
            "--sustained_max_loops=1",
            "--expected_business_gets=1",
            "--inprocess_pressure=true",
        ):
            self.assertIn(expected, args)
        for container in (wrapper, sidecar):
            mounts = {item["name"]: item for item in container["volumeMounts"]}
            self.assertEqual(
                "__RANK_KVC_RUNTIME_POD_DIR__", mounts["rank-kvc-runtime"]["mountPath"]
            )

    def test_rank_deploy_inherits_datasystem_arm_runtime(self):
        deploy = (
            ROOT / "scripts/deploy_deepfm_rank_burst_worker1.sh"
        ).read_text()
        for token in (
            'INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"',
            'INFERENCE_RUNTIME_ENV="$OUTPUT_DIR/inference-runtime.env"',
            "-o jsonpath='{.status.hostIP}'",
            "printf 'HOST_IP=%s\\n'",
            'name.startswith("DATASYSTEM_")',
            '"block_ds_consumer.so", "stub_gpu.so", "libabseil_dll.so"',
            '"libnvidia-ml.so" in token',
            'text.count(marker) != 2',
            'cat "/opt/pairec/lib/$library"',
            'rank_kvc_runtime_library=',
        ):
            self.assertIn(token, deploy)

        renderer = next(
            block for block in re.findall(r"<<'PY'\n(.*?)\nPY", deploy, re.DOTALL)
            if "required_preloads" in block
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "wrapper.yaml"
            runtime = root / "inference.env"
            runtime.write_text(
                "HOST_IP=141.61.91.188\n"
                "LD_LIBRARY_PATH=/datasystem/lib:/usr/local/lib\n"
                "LD_PRELOAD=/opt/block_ds_consumer.so /opt/stub_gpu.so "
                "/usr/lib/libnvidia-ml.so /ds/libabseil_dll.so\n"
                "DATASYSTEM_CLUSTER_NAME=pairec\n"
            )
            subprocess.run(
                [sys.executable, "-c", renderer,
                 str(ROOT / "k8s/deployment-deepfm-rank-burst-wrapper-worker1.yaml"),
                 str(output), "10.0.0.1:18211", "1", "0", str(runtime),
                 "/opt/pairec-rank-runtime", "/home/zcx/rank-kvc-runtime", ""],
                check=True, capture_output=True, text=True,
            )
            documents = list(yaml.safe_load_all(output.read_text()))
        containers = documents[0]["spec"]["template"]["spec"]["containers"]
        wrapper = next(item for item in containers if item["name"] == "rank-burst-wrapper")
        self.assertIn("--reverse_burst_endpoint=", wrapper["args"])
        for container_name in ("rank-burst-wrapper", "rank-kvc-burst-wrapper"):
            container = next(item for item in containers if item["name"] == container_name)
            env = {entry["name"]: entry["value"] for entry in container["env"]}
            self.assertEqual("141.61.91.188", env["HOST_IP"])
            self.assertEqual("pairec", env["DATASYSTEM_CLUSTER_NAME"])
            self.assertNotIn("libnvidia-ml.so", env["LD_PRELOAD"])
            self.assertIn(
                "/opt/pairec-rank-runtime/block_ds_consumer.so", env["LD_PRELOAD"]
            )

    def test_cpu_placement_diagnostics_do_not_abort_on_container_churn(self):
        deploy = (
            ROOT / "scripts/deploy_deepfm_rank_burst_worker1.sh"
        ).read_text()
        self.assertIn("len(containers)==expected", deploy)
        self.assertIn("text=True,capture_output=True", deploy)
        self.assertIn('"available":False', deploy)
        self.assertIn('"valid":not conflicts and not errors', deploy)
        self.assertIn('2>/dev/null || true', deploy)

    def test_full_combination_ab_wiring(self):
        benchmark = (
            ROOT / "scripts/benchmark_pairec_generation_kvc_rank_kvc_ab.sh"
        ).read_text()
        contention = (ROOT / "scripts/benchmark_brpc_kvc_contention.sh").read_text()
        combined = (
            ROOT / "scripts/validate_pairec_brpc_wrapper_kvc_combined.sh"
        ).read_text()
        for token in (
            "WRAPPER_CONCURRENCY=1000",
            "KVC_CONCURRENCY=32",
            "RANK_BURST_CONCURRENCY=1000",
            "RANK_KVC_CONCURRENCY=\"$rank_kvc_concurrency\"",
            "run_case rank_kvc_c1 1 0",
            "run_case rank_kvc_c32 32 4",
            "RANK_TIMEOUT_MS=1500",
        ):
            self.assertIn(token, benchmark)
        self.assertIn("arm_rank_kvc_burst", contention)
        self.assertIn("wait_rank_kvc_burst", contention)
        self.assertIn("refresh-and-arm", contention)
        for token in (
            'assert rank_kvc["business_get_count"] == 1',
            'assert rank_kvc["pressure_success"] == expected_rank_kvc_pressure',
            'assert rank_kvc["business_submit_rank"] == rank_kvc_concurrency',
            'assert rank_kvc["pressure_inflight_at_business_start"] == expected_rank_kvc_pressure',
            'assert rows[-1]["rank_kvc_business_get_ms"] <= rank_kvc_business_timeout_ms',
            'assert rows[-1]["rank_service_ms"] <= rank_kvc_service_timeout_ms',
        ):
            self.assertIn(token, combined)

    def test_ab_summary_delta(self):
        script = ROOT / "scripts/summarize_pairec_generation_kvc_rank_kvc_ab.py"
        spec = importlib.util.spec_from_file_location("rank_kvc_ab", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def case(concurrency, value):
            metrics = {
                name: {"avg": value, "p99": value * 2}
                for name in module.KEY_METRICS
            }
            return {
                "rank_kvc_enabled": True,
                "rank_kvc_concurrency": concurrency,
                "rank_kvc_object_size_bytes": 8388608,
                "rank_burst_concurrency": 1000,
                "response_semantic_fingerprints": ["stable-response"],
                "metrics": metrics,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            control = root / "control.json"
            treatment = root / "treatment.json"
            control.write_text(json.dumps(case(1, 10.0)))
            treatment.write_text(json.dumps(case(32, 25.0)))
            result = module.summarize(control, treatment)
        self.assertEqual(
            "PAIREC_GENERATION_C1000_KVC_C32_RANK_C1000_KVC_AB_OK",
            result["classification"],
        )
        self.assertEqual(15.0, result["comparison"]["client_e2e_ms"]["delta_avg"])
        self.assertTrue(result["semantic_match"])

    def test_ab_summary_rejects_semantic_change(self):
        script = ROOT / "scripts/summarize_pairec_generation_kvc_rank_kvc_ab.py"
        spec = importlib.util.spec_from_file_location("rank_kvc_ab_mismatch", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = {
            name: {"avg": 1.0, "p99": 1.0}
            for name in module.KEY_METRICS
        }
        base = {
            "rank_kvc_enabled": True,
            "rank_kvc_object_size_bytes": 8388608,
            "rank_burst_concurrency": 1000,
            "metrics": metrics,
        }
        control = dict(base, rank_kvc_concurrency=1,
                       response_semantic_fingerprints=["control"])
        treatment = dict(base, rank_kvc_concurrency=32,
                         response_semantic_fingerprints=["changed"])
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            left, right = root / "left.json", root / "right.json"
            left.write_text(json.dumps(control))
            right.write_text(json.dumps(treatment))
            with self.assertRaisesRegex(AssertionError, "changed recommendation semantics"):
                module.summarize(left, right)


if __name__ == "__main__":
    unittest.main()
