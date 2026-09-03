import os
import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "cpp" / "brpc_ub_probe"


class BrpcUbRecommendProbeTest(unittest.TestCase):
    def test_local_registry_wrapper_avoids_literal_remote_registries(self):
        script = (ROOT / "scripts" / "build_brpc_ub_recommend_probe_local_registry.sh").read_text()
        self.assertIn("modules/leveldb/1.23/MODULE.bazel", script)
        self.assertIn("OPENSSL_VERSION=${OPENSSL_VERSION:-3.3.2.bcr.1}", script)
        self.assertIn("modules/openssl/$OPENSSL_VERSION/MODULE.bazel", script)
        self.assertIn("RUN_MODULE_GRAPH=${RUN_MODULE_GRAPH:-0}", script)
        self.assertIn("BAZEL_LOCKFILE_MODE=update", script)
        self.assertIn("BAZEL_USE_PREINSTALLED_MAKE=1", script)
        self.assertIn("BRPC_GCC_TOOLSET_RUNTIME_OK", script)
        self.assertIn("BAZEL_ACTION_LD_LIBRARY_PATH", script)
        self.assertIn("--ignore_all_rc_files", script)
        self.assertIn("BRPC_LOCAL_REGISTRY_GRAPH_SKIPPED", script)
        self.assertNotIn("https://bcr.bazel.build", script)
        self.assertNotIn("raw.githubusercontent.com", script)
        build_script = (ROOT / "scripts" / "build_brpc_ub_recommend_probe.sh").read_text()
        self.assertIn("preinstalled_make_toolchain", build_script)
        self.assertIn("preinstalled_pkgconfig_toolchain", build_script)
        self.assertIn("--action_env=LD_LIBRARY_PATH", build_script)

    def test_known_good_builder_freezes_827_boringssl_baseline(self):
        script = (ROOT / "scripts" / "build_brpc_ub_recommend_known_good.sh").read_text()
        self.assertIn("827db2a9be6a3eac0a1ac3666b4a9cf33b976175", script)
        self.assertIn("9f80dc9fb5f06ba8b5997064c928b89bda266ffd", script)
        self.assertIn("BRPC_WITH_BORINGSSL=true", script)
        self.assertIn("BAZEL_LOCKFILE_MODE=${BAZEL_LOCKFILE_MODE:-off}", script)
        self.assertIn('"--ignore_all_rc_files"', script)
        self.assertNotIn("--announce_rc", script)
        self.assertIn("--define=BRPC_WITH_BORINGSSL=true", script)
        self.assertIn("BRPC_KNOWN_GOOD_RC_ISOLATED", script)
        self.assertIn("--repository_disable_download", script)
        self.assertIn("--registry=$secret_registry_uri", script)
        self.assertIn("restore_build_metadata", script)
        self.assertIn("BRPC_KNOWN_GOOD_METADATA_RESTORED", script)
        self.assertNotIn("3431fa24bace7ff0ee34c8717422a1905221ec02", script)

    def test_local_registry_wrapper_repairs_module_and_resolves_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            brpc_root = root / "brpc"
            bcr = root / "bcr"
            secret = root / "secret"
            fake_bin = root / "bin"
            brpc_root.mkdir()
            fake_bin.mkdir()
            (bcr / "bazel_registry.json").parent.mkdir(parents=True)
            (bcr / "bazel_registry.json").write_text("{}\n")
            foreign_cc_dir = bcr / "modules" / "rules_foreign_cc" / "0.12.0"
            foreign_cc_dir.mkdir(parents=True)
            (foreign_cc_dir / "MODULE.bazel").write_text('module(name = "rules_foreign_cc")\n')
            (foreign_cc_dir / "source.json").write_text("{}\n")
            for module, version in (("leveldb", "1.23"), ("openssl", "3.3.2.bcr.1")):
                module_dir = secret / "modules" / module / version
                module_dir.mkdir(parents=True)
                (module_dir / "MODULE.bazel").write_text(f'module(name = "{module}")\n')
                (module_dir / "source.json").write_text("{}\n")
            (secret / "bazel_registry.json").write_text("{}\n")
            (brpc_root / "MODULE.bazel").write_text(
                textwrap.dedent(
                    """
                    bazel_dep(name = 'openssl', version = '3.3.2')
                    single_version_override(
                        module_name = "leveldb",
                        registry = "[file:///bad](file:///bad)",
                    )
                    single_version_override(
                        module_name = "openssl",
                        version = "3.3.2.bcr.1",
                        registry = "[file:///bad](file:///bad)",
                    )
                    """
                )
            )
            fake_bazel = fake_bin / "bazel"
            fake_bazel.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \" $* \" == *\" mod graph \"* ]]; then "
                "printf '%s\\n' \"$PWD\" >\"$BAZEL_CWD_LOG\"; echo 'brpc@1.15.0'; fi\n"
            )
            fake_bazel.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "BRPC_ROOT": str(brpc_root),
                    "LOCAL_BCR_REGISTRY": str(bcr),
                    "LOCAL_SECRET_REGISTRY": str(secret),
                    "BAZEL_OUTPUT_BASE": str(root / "output-base"),
                    "MODULE_GRAPH_OUT": str(root / "graph.txt"),
                    "BAZEL_CWD_LOG": str(root / "bazel-cwd.txt"),
                    "RUN_MODULE_GRAPH": "1",
                    "RUN_BUILD": "0",
                }
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "build_brpc_ub_recommend_probe_local_registry.sh")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            module_text = (brpc_root / "MODULE.bazel").read_text()
            self.assertIn("BRPC_LOCAL_REGISTRY_GRAPH_OK", result.stdout)
            self.assertIn("BRPC_LOCAL_REGISTRY_RC_ISOLATION_OK", result.stdout)
            self.assertEqual(module_text.count(secret.resolve().as_uri()), 2)
            self.assertIn("bazel_dep(name = 'openssl', version = \"3.3.2.bcr.1\")", module_text)
            self.assertEqual(
                module_text.count('bazel_dep(name = "rules_foreign_cc", version = "0.12.0")'), 1
            )
            self.assertNotIn("[file:", module_text)
            self.assertEqual((root / "graph.txt").read_text().strip(), "brpc@1.15.0")
            self.assertEqual((root / "bazel-cwd.txt").read_text().strip(), str(brpc_root))

    def test_transport_is_explicitly_enabled_on_both_ends(self):
        server = (PROBE_DIR / "minimal_recommend_server.cpp").read_text()
        client = (PROBE_DIR / "minimal_recommend_client.cpp").read_text()
        self.assertIn("options.use_ub = FLAGS_ubsocket_use_ub", server)
        self.assertIn("options.use_ub = FLAGS_ubsocket_use_ub", client)
        self.assertIn("response_attachment().append(payload)", server)
        self.assertIn("echoedPayload != payload", client)
        self.assertIn("options.connect_timeout_ms = FLAGS_probe_connect_timeout_ms", client)
        self.assertIn('DEFINE_string(\n    probe_connection_type,\n    "",', client)
        self.assertIn("options.connection_type = FLAGS_probe_connection_type", client)

        for script_name in ("run_brpc_ub_recommend_server.sh", "run_brpc_ub_recommend_matrix.sh"):
            script = (ROOT / "scripts" / script_name).read_text()
            self.assertIn("--ubsocket_backup_link_enable=false", script)
            self.assertIn("--ubsocket_degrade_enable=false", script)
            self.assertIn("URMA_RUNTIME_LIB_DIR=${URMA_RUNTIME_LIB_DIR:-/usr/lib64}", script)
            self.assertIn('export LD_LIBRARY_PATH="$URMA_RUNTIME_LD_LIBRARY_PATH', script)

    def test_echo_baseline_script_uses_same_offline_ub_build(self):
        script = (ROOT / "scripts" / "verify_brpc_ub_echo_baseline.sh").read_text()
        self.assertIn("ACTION=${ACTION:-client}", script)
        self.assertIn("//example:echo_c++_server", script)
        self.assertIn("//example:echo_c++_client", script)
        self.assertIn("--define brpc_with_urma=true", script)
        self.assertIn("--ignore_all_rc_files", script)
        self.assertIn("build_brpc_ub_recommend_probe_local_registry.sh", script)
        self.assertIn("--ubsocket_backup_link_enable=false", script)
        self.assertIn("--ubsocket_degrade_enable=false", script)
        self.assertIn("invalid bthread_key", script)
        self.assertIn("BRPC_UB_ECHO_BASELINE_PASS", script)
        self.assertIn("BRPC_UB_ECHO_BASELINE_CRASH", script)

    def test_echo_baseline_client_classifies_pass_and_known_crash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            runtime_dir = root / "lib64"
            provider_dir = runtime_dir / "urma"
            provider_dir.mkdir(parents=True)
            (runtime_dir / "liburma.so").write_text("")
            client = root / "echo_client"
            client.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'bind jetty success'\n"
                "echo 'Received response from test: hello world'\n"
            )
            client.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "ACTION": "client",
                    "CLIENT_BIN": str(client),
                    "LOG_DIR": str(root / "logs"),
                    "RUN_SECONDS": "1",
                    "URMA_RUNTIME_LIB_DIR": str(runtime_dir),
                    "URMA_PROVIDER_LIB_DIR": str(provider_dir),
                    "URMA_RUNTIME_LD_LIBRARY_PATH": f"{runtime_dir}:{provider_dir}",
                }
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "verify_brpc_ub_echo_baseline.sh")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("BRPC_UB_ECHO_BASELINE_PASS responses=1", result.stdout)

            client.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'bthread_setspecific is called on invalid bthread_key_t'\n"
                "exit 139\n"
            )
            client.chmod(0o755)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "verify_brpc_ub_echo_baseline.sh")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("BRPC_UB_ECHO_BASELINE_CRASH", result.stderr)

    def test_bthread_key_diagnostic_uses_local_source_of_truth(self):
        script = (ROOT / "scripts" / "diagnose_brpc_ub_bthread_key_crash.sh").read_text()
        self.assertIn("source_of_truth=local_brpc_and_bazel_external_ubsocket", script)
        self.assertIn("public_repository_assumption=disabled", script)
        self.assertIn("bthread_(key_create2?|key_delete|setspecific|getspecific)", script)
        self.assertIn("initialization-order.txt", script)
        self.assertIn("addr2line -Cfipe", script)
        self.assertIn("objdump -dC", script)
        self.assertIn("client-coredump-info.txt", script)
        self.assertIn("grep -RniE -C 5", script)
        self.assertNotIn("for command in rg ", script)
        self.assertNotIn("EXPECTED_BRPC_COMMIT", script)
        self.assertNotIn("gitcode.com", script)
        self.assertNotIn("atomgit.com", script)

    def test_bthread_key_diagnostic_runs_with_local_only_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            brpc_root = root / "brpc"
            for source_dir in ("src/brpc", "src/bthread", "src/butil"):
                (brpc_root / source_dir).mkdir(parents=True, exist_ok=True)
            (brpc_root / "src/brpc/example.cpp").write_text(
                "bthread_key_t key; bthread_key_create(&key, nullptr);\n"
            )
            ubs_root = root / "output-base" / "external" / "_main~local_deps~ubsocket"
            (ubs_root / "src/ubsocket").mkdir(parents=True)
            (ubs_root / "src/ubsocket/example.c").write_text("bthread_setspecific(key, data);\n")
            client_log = root / "client.log"
            client_log.write_text(
                "bthread_setspecific is called on invalid bthread_key_t{index=0 version=0}\n"
                "#0 0x0000000000000000 bthread::KeyTable::set_data()\n"
            )
            output_dir = root / "evidence"
            env = os.environ.copy()
            env.update(
                {
                    "BRPC_ROOT": str(brpc_root),
                    "BAZEL_OUTPUT_BASE": str(root / "output-base"),
                    "CLIENT_BIN": "/bin/true",
                    "SERVER_BIN": "/bin/true",
                    "CLIENT_LOG": str(client_log),
                    "SERVER_LOG": str(root / "missing-server.log"),
                    "OUTPUT_DIR": str(output_dir),
                }
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "diagnose_brpc_ub_bthread_key_crash.sh")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("BRPC_UB_BTHREAD_KEY_DIAGNOSTIC_OK", result.stdout)
            self.assertIn(
                "classification=BRPC_UB_BTHREAD_KEY_CRASH_CONFIRMED",
                (output_dir / "summary.txt").read_text(),
            )
            self.assertIn("bthread_key_create", (output_dir / "bthread-key-usage.txt").read_text())
            self.assertTrue(pathlib.Path(f"{output_dir}.tar.gz").is_file())

    def test_bthread_trace_key_fix_allocates_keys_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            brpc_root = root / "brpc"
            bthread_dir = brpc_root / "src" / "bthread"
            brpc_dir = brpc_root / "src" / "brpc"
            bthread_dir.mkdir(parents=True)
            brpc_dir.mkdir(parents=True)
            (bthread_dir / "bthread.cpp").write_text(
                textwrap.dedent(
                    """
                    #ifdef BRPC_WITH_URMA
                    bthread_key_t ubsocket_trace_rpcid_key{0, 0};
                    bthread_key_t ubsocket_trace_call_timestamp{1, 0};
                    #endif
                    namespace bthread {
                    """
                ).lstrip("\n")
            )
            (brpc_dir / "ubsocket_initializer.cpp").write_text(
                "    }\n"
                "    ubsocket_set_log_level(ub_log_level);\n"
                "    ubsocket_set_logger(UBSocketLogger);\n"
                "    /* initialize ubsocket */\n"
                "    u_init_options_t options;\n"
            )
            env = os.environ.copy()
            env.update(
                {
                    "BRPC_ROOT": str(brpc_root),
                    "BACKUP_ROOT": str(root / "backups"),
                }
            )
            command = ["bash", str(ROOT / "scripts" / "apply_brpc_ub_trace_key_fix.sh")]
            first = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True, check=False
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn("BRPC_UB_TRACE_KEY_FIX_APPLIED", first.stdout)
            bthread_text = (bthread_dir / "bthread.cpp").read_text()
            initializer_text = (brpc_dir / "ubsocket_initializer.cpp").read_text()
            self.assertIn(
                "ubsocket_trace_rpcid_key = INVALID_BTHREAD_KEY", bthread_text
            )
            self.assertIn(
                "bthread_key_create(&ubsocket_trace_rpcid_key, NULL)", initializer_text
            )
            self.assertIn("bthread_key_delete(ubsocket_trace_rpcid_key)", initializer_text)
            self.assertEqual(len(list((root / "backups").glob("*/src/bthread/bthread.cpp"))), 1)

            second = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True, check=True
            )
            self.assertIn("BRPC_UB_TRACE_KEY_FIX_ALREADY_APPLIED", second.stdout)
            self.assertEqual(len(list((root / "backups").glob("*/src/bthread/bthread.cpp"))), 1)

    def test_payload_matrix_contains_boundary_and_large_values(self):
        client = (PROBE_DIR / "minimal_recommend_client.cpp").read_text()
        self.assertIn("0,1,4096,4097,65536,1048576,3670016", client)
        self.assertIn("Sha256Hex(payload)", client)
        self.assertIn("actualMetadata != expectedMetadata", client)

    def test_bazel_package_reuses_brpc_target(self):
        build = (PROBE_DIR / "BUILD.bazel").read_text()
        client = (PROBE_DIR / "minimal_recommend_client.cpp").read_text()
        server = (PROBE_DIR / "minimal_recommend_server.cpp").read_text()
        self.assertIn('"//:brpc"', build)
        self.assertIn('"-std=c++17"', build)
        self.assertNotIn("datasystem", build.lower())
        self.assertNotIn("trt", build.lower())
        self.assertIn('#include "pairec_ub_probe/recommend.pb.h"', client)
        self.assertIn('#include "pairec_ub_probe/recommend.pb.h"', server)

    def test_build_script_stages_complete_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update({"BRPC_ROOT": temp_dir, "SKIP_VERSION_CHECK": "1", "STAGE_ONLY": "1"})
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "build_brpc_ub_recommend_probe.sh")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            package = pathlib.Path(temp_dir) / "pairec_ub_probe"
            self.assertIn("BRPC_UB_RECOMMEND_STAGE_OK", result.stdout)
            self.assertEqual(
                {path.name for path in package.iterdir()},
                {
                    "BUILD.bazel",
                    "minimal_recommend_server.cpp",
                    "minimal_recommend_client.cpp",
                    "payload_integrity.h",
                    "recommend.proto",
                },
            )

    @unittest.skipUnless(shutil.which("g++"), "g++ is required")
    def test_payload_helper_compiles_and_hashes(self):
        source = textwrap.dedent(
            """
            #include "payload_integrity.h"
            #include <iostream>

            int main()
            {
                const std::string payload = pairec::brpc_ub_probe::BuildDeterministicPayload(4097, 7);
                const std::string digest = pairec::brpc_ub_probe::Sha256Hex(payload);
                std::cout << payload.size() << " " << digest.size() << "\\n";
                return payload.size() == 4097 && digest.size() == 64 ? 0 : 1;
            }
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = pathlib.Path(temp_dir) / "payload_smoke.cpp"
            binary_path = pathlib.Path(temp_dir) / "payload_smoke"
            source_path.write_text(source)
            compile_result = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-I",
                    str(PROBE_DIR),
                    str(source_path),
                    "-lcrypto",
                    "-o",
                    str(binary_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            result = subprocess.run([str(binary_path)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "4097 64")


if __name__ == "__main__":
    unittest.main()
