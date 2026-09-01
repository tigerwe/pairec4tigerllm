from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_kvc_ub_integrity_probe.sh"
SOURCE = ROOT / "cpp" / "kvc_burst" / "kvc_ub_integrity_probe.cpp"


class KvcUbIntegrityProbeTest(unittest.TestCase):
    def _write_fake_probe(self, directory: pathlib.Path, tcp: bool) -> pathlib.Path:
        path = directory / "fake_probe.sh"
        path.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail
                prefix=
                iterations=0
                while (($#)); do
                  case "$1" in
                    --prefix) prefix=$2; shift 2 ;;
                    --iterations) iterations=$2; shift 2 ;;
                    *) shift ;;
                  esac
                done
                log="$DATASYSTEM_CLIENT_LOG_DIR/ds_client_access.log"
                for ((i=0; i<iterations; i++)); do
                  echo "DS_KV_CLIENT_SET Object_key:${{prefix}}_${{i}},transportType:UB," >>"$log"
                  transport=UB
                  if [[ "{str(tcp).lower()}" == true && "$i" == 0 ]]; then transport=TCP; fi
                  echo "DS_KV_CLIENT_GET Object_key:${{prefix}}_${{i}},transportType:${{transport}}," >>"$log"
                done
                echo KVC_UB_INTEGRITY_PASS
                """
            ),
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def _run(self, tcp: bool) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            probe = self._write_fake_probe(root, tcp)
            env = os.environ.copy()
            env.update(
                {
                    "KVC_UB_PROBE_BIN": str(probe),
                    "LOG_DIR": str(root / "logs"),
                    "OUTPUT_DIR": str(root / "evidence"),
                    "PREFIX": "KvcUbRunnerTest",
                    "ITERATIONS": "3",
                }
            )
            return subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def test_source_contains_required_integrity_checks(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("enableCrossNodeConnection = true", text)
        self.assertIn("SHA256(", text)
        self.assertIn("std::memcmp", text)
        self.assertIn("actualSize == config.objectSize", text)
        self.assertIn("KVC_UB_INTEGRITY_PASS", text)

    def test_runner_accepts_only_complete_ub_evidence(self):
        result = self._run(tcp=False)
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertIn("set_ub=3 get_ub=3 tcp=0", result.stdout)
        self.assertIn("KVC_UB_POC_PASS", result.stdout)

    def test_runner_rejects_tcp_fallback(self):
        result = self._run(tcp=True)
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("expected 3 UB Get records, observed 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
