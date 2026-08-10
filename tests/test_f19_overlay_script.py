import json
import pathlib
import re
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_f19_datasystem_attribution_overlay.sh"


class F19OverlayScriptTest(unittest.TestCase):
    def embedded_python_blocks(self):
        text = SCRIPT.read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
        self.assertEqual(2, len(blocks))
        return blocks

    def test_embedded_python_is_valid(self):
        for index, block in enumerate(self.embedded_python_blocks(), start=1):
            compile(block, f"embedded-python-{index}", "exec")

    def test_generated_patch_has_expected_overlay_contract(self):
        block = self.embedded_python_blocks()[1]
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "patch.json"
            original_argv = sys.argv
            try:
                sys.argv = [
                    "embedded-python",
                    "brpc-inference",
                    "/home/zcx/pairec-f19-runtime",
                    "/opt/pairec-f19",
                    "/TensorRT-LLM/cpp/build/tensorrt_llm/libtensorrt_llm.so",
                    "/opt/pairec-f19/lib:/existing/lib",
                    str(output),
                ]
                exec(compile(block, "embedded-python-patch", "exec"), {})
            finally:
                sys.argv = original_argv

            patch = json.loads(output.read_text())
            pod_spec = patch["spec"]["template"]["spec"]
            container = pod_spec["containers"][0]
            self.assertEqual(
                ["/opt/pairec-f19/bin/brpc_inference_server"],
                container["command"],
            )
            self.assertEqual(
                {"f19-runtime-bin", "f19-runtime-lib", "f19-trtllm-file"},
                {volume["name"] for volume in pod_spec["volumes"]},
            )
            trtllm_mount = next(
                mount for mount in container["volumeMounts"]
                if mount["name"] == "f19-trtllm-file"
            )
            self.assertEqual(
                "/TensorRT-LLM/cpp/build/tensorrt_llm/libtensorrt_llm.so",
                trtllm_mount["mountPath"],
            )
            trtllm_volume = next(
                volume for volume in pod_spec["volumes"]
                if volume["name"] == "f19-trtllm-file"
            )
            self.assertEqual("File", trtllm_volume["hostPath"]["type"])


if __name__ == "__main__":
    unittest.main()
