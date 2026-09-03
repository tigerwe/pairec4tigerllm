#!/usr/bin/env bash
set -euo pipefail

KVC_CLIENT_HOST="${KVC_CLIENT_HOST:-node1}"
KVC_REMOTE_REPO="${KVC_REMOTE_REPO:-/home/zcx/workspace/pairec4tigerllm}"
KVC_WORKER_HOST="${KVC_WORKER_HOST:-141.62.33.105}"
KVC_WORKER_PORT="${KVC_WORKER_PORT:-32501}"
KVC_ITERATIONS="${KVC_ITERATIONS:-10}"
POST_RANK_SERVER="${POST_RANK_SERVER:-127.0.0.1:18311}"
POST_RANK_REQUESTS="${POST_RANK_REQUESTS:-3}"
HOP1_LOG="${HOP1_LOG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/brpc-ub-post-rank-kvc-joint/$(date +%Y%m%d-%H%M%S)}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-60}"

die() { echo "ERROR: $*" >&2; exit 1; }
for command in ssh grep; do command -v "$command" >/dev/null || die "missing command: $command"; done
[[ "$KVC_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || die "KVC_ITERATIONS must be positive"
[[ "$POST_RANK_REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "POST_RANK_REQUESTS must be positive"
[[ -n "$HOP1_LOG" && -f "$HOP1_LOG" ]] || die "HOP1_LOG must name the active UB Hop1 log"
grep -Fq '"role":"hop1","transport":"ub"' "$HOP1_LOG" \
  || die "Hop1 log does not identify an active UB process"
mkdir -p "$OUTPUT_DIR"

run_joint_case() {
  local label="$1" object_size="$2" case_dir="$OUTPUT_DIR/$label"
  local run_id start_file post_rank_start_file remote_output kvc_console
  local post_rank_console kvc_pid post_rank_pid deadline
  run_id="$(date +%Y%m%d-%H%M%S)-$$-$label"
  start_file="/tmp/pairec-kvc-joint-start-$run_id"
  post_rank_start_file="/tmp/pairec-post-rank-joint-start-$run_id"
  remote_output="/root/kvc-ub-joint/$run_id"
  kvc_console="$case_dir/kvc.console.log"
  post_rank_console="$case_dir/post-rank.console.log"
  mkdir -p "$case_dir"
  rm -f "$post_rank_start_file"

  ssh "$KVC_CLIENT_HOST" rm -f "$start_file"
  ssh "$KVC_CLIENT_HOST" env \
    WORKER_HOST="$KVC_WORKER_HOST" WORKER_PORT="$KVC_WORKER_PORT" \
    CONCURRENCY=32 ITERATIONS="$KVC_ITERATIONS" OBJECT_SIZE="$object_size" \
    PREFIX="PairecJoint_${run_id}" START_FILE="$start_file" \
    LOG_DIR="$remote_output/client-log" OUTPUT_DIR="$remote_output/evidence" \
    bash "$KVC_REMOTE_REPO/scripts/run_kvc_ub_integrity_probe.sh" \
    >"$kvc_console" 2>&1 &
  kvc_pid=$!

  deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  until grep -Fq "KVC_UB_CONCURRENT_READY concurrency=32 object_size=$object_size" "$kvc_console"; do
    if ! kill -0 "$kvc_pid" 2>/dev/null; then
      wait "$kvc_pid" || true
      die "remote KVC c32 process exited before ready: $kvc_console"
    fi
    (( SECONDS < deadline )) || die "timed out waiting for remote KVC c32 readiness"
    sleep 1
  done

  env SERVER="$POST_RANK_SERVER" TRANSPORT=ub REQUESTS="$POST_RANK_REQUESTS" \
    HOP1_LOG="$HOP1_LOG" EVIDENCE_DIR="$case_dir/post-rank-evidence" \
    START_FILE="$post_rank_start_file" START_WAIT_TIMEOUT_MS=60000 \
    bash scripts/run_brpc_ub_post_rank_qualification.sh \
    >"$post_rank_console" 2>&1 &
  post_rank_pid=$!

  deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  until grep -Fq "POST_RANK_QUALIFICATION_READY" "$post_rank_console"; do
    if ! kill -0 "$post_rank_pid" 2>/dev/null; then
      wait "$post_rank_pid" || true
      wait "$kvc_pid" || true
      die "post-rank qualification exited before ready: $post_rank_console"
    fi
    (( SECONDS < deadline )) || die "timed out waiting for post-rank readiness"
    sleep 1
  done

  # Both clients are fully initialized and blocked on their gates. Release them
  # back-to-back so the measured c1000 and c32 work actually overlaps.
  ssh "$KVC_CLIENT_HOST" touch "$start_file"
  touch "$post_rank_start_file"
  wait "$post_rank_pid" || {
    wait "$kvc_pid" || true
    die "post-rank UB qualification failed while KVC c32 was active"
  }
  wait "$kvc_pid" || die "remote KVC c32 qualification failed: $kvc_console"
  ssh "$KVC_CLIENT_HOST" rm -f "$start_file"
  rm -f "$post_rank_start_file"

  grep -Fq "KVC_UB_ACCESS_LOG_PASS" "$kvc_console" \
    || die "KVC access-log proof missing: $kvc_console"
  grep -Fq "concurrency=32 object_size=$object_size" "$kvc_console" \
    || die "KVC c32/object-size proof missing: $kvc_console"
  grep -Fq 'tcp=0' "$kvc_console" || die "KVC TCP fallback was observed"
  grep -Fq 'POST_RANK_QUALIFICATION_START_RELEASED' "$post_rank_console" \
    || die "post-rank start-gate release proof missing"
  grep -Fq 'KVC_UB_CONCURRENT_START_RELEASED' "$kvc_console" \
    || die "KVC start-gate release proof missing"
  grep -Fq 'PAIREC_POST_RANK_UB_C1000_100K_OK' "$post_rank_console" \
    || die "post-rank UB proof missing"
  echo "JOINT_CASE_PASS label=$label kvc_concurrency=32 kvc_object_size=$object_size post_rank_concurrency=1000 post_rank_payload_bytes=102400"
}

run_joint_case generation_3_5mib 3670016
run_joint_case rank_8mib 8388608

echo "PAIREC_BRPC_UB_C1000_KVC_UB_C32_JOINT_PASS"
echo "output_dir=$OUTPUT_DIR"
