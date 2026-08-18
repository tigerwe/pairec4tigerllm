#!/usr/bin/env bash
# Diagnose a failed BRPC Wrapper + KVC matrix run from saved artifacts only.
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-${1:-}}
ARM=${ARM:-combined}
ROUND=${ROUND:-all}
OUTPUT_DIR=${OUTPUT_DIR:-}

die() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  echo "usage: RUN_ROOT=<matrix-root> [ARM=combined] [ROUND=all|N] $0"
  echo "       $0 <matrix-root>"
}
if [[ "$RUN_ROOT" == "-h" || "$RUN_ROOT" == "--help" ]]; then
  usage
  exit 0
fi
command -v python3 >/dev/null 2>&1 || die "missing command: python3"

if [[ -z "$RUN_ROOT" ]]; then
  shopt -s nullglob
  candidates=(/tmp/pairec-brpc-wrapper-kvc-matrix/*)
  shopt -u nullglob
  [[ ${#candidates[@]} -gt 0 ]] || die "no matrix run found under /tmp/pairec-brpc-wrapper-kvc-matrix"
  RUN_ROOT=$(ls -dt "${candidates[@]}" | head -1)
fi
[[ -d "$RUN_ROOT" ]] || die "RUN_ROOT does not exist: $RUN_ROOT"

if [[ -d "$RUN_ROOT/$ARM/contention" ]]; then
  CONTENTION_DIR="$RUN_ROOT/$ARM/contention"
elif [[ $(basename "$RUN_ROOT") == contention ]]; then
  CONTENTION_DIR="$RUN_ROOT"
else
  die "cannot find contention artifacts: $RUN_ROOT/$ARM/contention"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$RUN_ROOT/diagnostic-${ARM}-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

python3 - "$RUN_ROOT" "$CONTENTION_DIR" "$ROUND" "$OUTPUT_DIR" <<'PY' \
  | tee "$OUTPUT_DIR/summary.txt"
import json
import pathlib
import re
import sys

run_root = pathlib.Path(sys.argv[1])
contention = pathlib.Path(sys.argv[2])
requested_round = sys.argv[3]
output = pathlib.Path(sys.argv[4])


def read_text(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def json_events(path, event_name=None):
    events = []
    for line in read_text(path).splitlines():
        pos = line.find("{")
        if pos < 0:
            continue
        try:
            event = json.loads(line[pos:])
        except json.JSONDecodeError:
            continue
        if event_name is None or event.get("event") == event_name:
            events.append(event)
    return events


def integer_file(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def round_number(path):
    match = re.fullmatch(r"round-(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


result = read_json(contention / "result.json") or {}
expected_repeats = int(result.get("expected_repeats", 0) or 0)
result_rows = {str(row.get("round")): row for row in result.get("rows", [])}
round_dirs = sorted(
    (path for path in contention.glob("round-*") if path.is_dir()),
    key=round_number,
)
if requested_round != "all":
    wanted = requested_round.removeprefix("round-")
    round_dirs = [path for path in round_dirs if path.name == f"round-{wanted}"]
    if not round_dirs:
        raise SystemExit(f"ERROR: round not found: {requested_round}")


def replay_attempts(round_dir):
    attempts = []
    archived = sorted(
        (path for path in round_dir.glob("replay-attempt-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    attempts.extend(archived)
    final = round_dir / "replay"
    if final.is_dir():
        attempts.append(final)
    return attempts


def inspect_attempt(path):
    client = read_json(path / "client.json") or {}
    response = read_json(path / "response.json") or {}
    trace = read_json(path / "summary.json") or {}
    request_id = (
        client.get("request_id")
        or response.get("request_id")
        or trace.get("request_id")
        or ""
    )
    native = json_events(path / "brpc_trtllm.log", "datasystem_request_complete")
    if request_id:
        matching = [event for event in native if event.get("request_id") == request_id]
        if matching:
            native = matching
    completion = native[-1] if native else trace.get("datasystem_request_complete") or {}
    kvc = json_events(path / "kvc_burst.log")
    if not kvc and isinstance(trace.get("kvc_proxy_events"), list):
        kvc = trace["kvc_proxy_events"]
    if request_id:
        matching = [event for event in kvc if event.get("request_id") == request_id]
        if matching:
            kvc = matching
    starts = [event for event in kvc if event.get("event") == "kvc_burst_start"]
    completes = [event for event in kvc if event.get("event") in {"kvc_burst_complete", "kvc_burst_result"}]
    console = read_text(path.parent / f"{path.name}.console.log")
    if path.name == "replay":
        console = read_text(path.parent / "replay.console.log")
    return {
        "path": str(path),
        "request_id": request_id,
        "client_ok": client.get("ok"),
        "response_code": client.get("response_code", response.get("code")),
        "get_count": completion.get("get_count"),
        "set_count": completion.get("set_count"),
        "attribution_complete": completion.get("attribution_complete"),
        "kvc_start_found": bool(starts),
        "kvc_complete_found": bool(completes),
        "kvc_complete": completes[-1] if completes else None,
        "console_error": next(
            (line.strip() for line in console.splitlines() if "ERROR:" in line or "AssertionError" in line),
            "",
        ),
    }


def classify(round_dir, attempts, result_row):
    replay_exit = integer_file(round_dir / "replay.exit_code")
    arm_exit = integer_file(round_dir / "kvc-burst-arm.exit_code")
    rearm_exit = integer_file(round_dir / "kvc-burst-arm-retry.exit_code")
    if arm_exit not in (None, 0) or rearm_exit not in (None, 0):
        return "KVC_ARM_OR_REARM_FAILURE"
    if not attempts:
        return "REPLAY_ARTIFACT_MISSING"
    last = attempts[-1]
    if last["response_code"] not in (None, 200) or last["client_ok"] is False:
        return "BUSINESS_RESPONSE_FAILURE"
    complete = last.get("kvc_complete") or {}
    if int(complete.get("sustained_errors", 0) or 0) > 0:
        return "SUSTAINED_PRESSURE_GET_FAILURE"
    if last.get("get_count") == 0:
        return "ZERO_ONBOARD_GET_RETRY_EXHAUSTED"
    if last.get("get_count") in (1, 2) and not last.get("kvc_complete_found"):
        return "KVC_TRIGGER_OR_COMPLETION_MISSING"
    if replay_exit not in (None, 0):
        return "REPLAY_POSTPROCESS_OR_VALIDATION_FAILURE"
    if result_row and result_row.get("valid") is False:
        if not result_row.get("runtime_ok", True):
            return "POD_RESTART_OR_CRASH"
        if not result_row.get("exact_attribution_ok", True):
            return "DATASYSTEM_ATTRIBUTION_FAILURE"
        if not result_row.get("pressure_ok", True):
            return "PRESSURE_GATE_FAILURE"
        return "ROUND_VALIDATION_FAILURE"
    if last.get("kvc_complete_found") and replay_exit in (None, 0):
        return "ROUND_VALID"
    return "UNKNOWN_ROUND_FAILURE"


rounds = []
for round_dir in round_dirs:
    number = str(round_number(round_dir))
    attempts = [inspect_attempt(path) for path in replay_attempts(round_dir)]
    row = result_rows.get(number)
    worker_metrics = read_json(round_dir / "datasystem-worker-metrics.json")
    worker_error = read_text(round_dir / "datasystem-worker-metrics.error").strip()
    prime_dirs = sorted(str(path) for path in round_dir.glob("replay-prime-*") if path.is_dir())
    item = {
        "round": int(number),
        "classification": classify(round_dir, attempts, row),
        "replay_attempts_recorded": integer_file(round_dir / "replay.attempts"),
        "replay_exit_code": integer_file(round_dir / "replay.exit_code"),
        "arm_exit_code": integer_file(round_dir / "kvc-burst-arm.exit_code"),
        "rearm_exit_code": integer_file(round_dir / "kvc-burst-arm-retry.exit_code"),
        "focused_prime_dirs": prime_dirs,
        "attempts": attempts,
        "result_row": row,
        "worker_metrics": worker_metrics,
        "worker_metrics_error": worker_error or None,
        "control_log": read_text(round_dir / "kvc-burst-control.log").strip(),
    }
    rounds.append(item)

failures = [item for item in rounds if item["classification"] != "ROUND_VALID"]
classes = [item["classification"] for item in failures]
if not failures:
    overall = "NO_FAILED_ROUND_FOUND"
elif len(set(classes)) == 1:
    overall = classes[0]
else:
    overall = "MULTIPLE_FAILURE_CLASSES"

diagnosis = {
    "classification": overall,
    "run_root": str(run_root),
    "contention_dir": str(contention),
    "contention_status": result.get("status"),
    "expected_repeats": expected_repeats,
    "valid_repeats": result.get("valid_repeats"),
    "rounds": rounds,
}
(output / "summary.json").write_text(json.dumps(diagnosis, indent=2) + "\n")

print("PaiRec BRPC Wrapper/KVC matrix failure diagnosis")
print(f"  classification={overall}")
print(f"  run_root={run_root}")
print(f"  contention={result.get('status', 'missing')} valid={result.get('valid_repeats', 'unknown')}/{expected_repeats or 'unknown'}")
print("  round class attempts replay_exit request_id response get set kvc_start kvc_complete sustained_loops sustained_errors")
for item in rounds:
    last = item["attempts"][-1] if item["attempts"] else {}
    complete = last.get("kvc_complete") or {}
    print(
        f"  {item['round']:>5} {item['classification']:<38} "
        f"{item['replay_attempts_recorded'] or len(item['attempts']):>8} "
        f"{str(item['replay_exit_code']):>11} "
        f"{last.get('request_id', '-') or '-'} "
        f"{str(last.get('response_code')):>8} "
        f"{str(last.get('get_count')):>3} {str(last.get('set_count')):>3} "
        f"{str(last.get('kvc_start_found')):>9} {str(last.get('kvc_complete_found')):>12} "
        f"{str(complete.get('sustained_loop_gets', 0)):>15} "
        f"{str(complete.get('sustained_errors', 0)):>16}"
    )
    for index, attempt in enumerate(item["attempts"], 1):
        print(
            f"        attempt={index} request_id={attempt['request_id'] or '-'} "
            f"response={attempt['response_code']} get={attempt['get_count']} "
            f"set={attempt['set_count']} start={attempt['kvc_start_found']} "
            f"complete={attempt['kvc_complete_found']} error={attempt['console_error'] or 'none'}"
        )
    if item["focused_prime_dirs"]:
        print(f"        focused_prime_count={len(item['focused_prime_dirs'])}")
    if item["worker_metrics_error"]:
        print(f"        worker_metrics_error={item['worker_metrics_error']}")

print(f"  summary_json={output / 'summary.json'}")
print(f"  output_dir={output}")
print("PAIREC_BRPC_WRAPPER_KVC_MATRIX_DIAGNOSTIC_COMPLETE")
PY
