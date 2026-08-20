#!/usr/bin/env bash
# Recover the inference Deployment after an interrupted BRPC/KVC matrix run.
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
DEPLOYMENT=${DEPLOYMENT:-inference-brpc-trtllm}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-10m}
PROCESS_STOP_TIMEOUT_SECONDS=${PROCESS_STOP_TIMEOUT_SECONDS:-600}
BACKUP_FILE=${BACKUP_FILE:-/tmp/f14-kvc-burst-recovery-backup.json}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

die() { echo "ERROR: $*" >&2; exit 1; }

command -v kubectl >/dev/null 2>&1 || die "kubectl is required"
command -v pgrep >/dev/null 2>&1 || die "pgrep is required"
[[ "$PROCESS_STOP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || die "PROCESS_STOP_TIMEOUT_SECONDS must be positive"

declare -A selected=()
patterns=(
  'scripts/benchmark_pairec_brpc_wrapper_kvc_matrix.sh'
  'scripts/validate_pairec_brpc_wrapper_kvc_combined.sh'
  'scripts/benchmark_brpc_kvc_contention.sh'
)

for pattern in "${patterns[@]}"; do
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [ "$pid" -ne "$$" ] || continue
    [ "$pid" -ne "$PPID" ] || continue
    selected["$pid"]=1
  done < <(pgrep -f -- "$pattern" || true)
done

if [ "${#selected[@]}" -gt 0 ]; then
  echo "== Resume and terminate interrupted matrix processes =="
  mapfile -t pids < <(printf '%s\n' "${!selected[@]}" | sort -n)
  ps -o pid=,ppid=,stat=,args= -p "$(IFS=,; echo "${pids[*]}")" || true
  for pid in "${pids[@]}"; do
    kill -CONT "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  deadline=$((SECONDS + PROCESS_STOP_TIMEOUT_SECONDS))
  while true; do
    remaining=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        remaining+=("$pid")
      fi
    done
    [ "${#remaining[@]}" -gt 0 ] || break
    if (( SECONDS >= deadline )); then
      echo "ERROR: interrupted processes did not exit after TERM: ${remaining[*]}" >&2
      echo "Refusing to use SIGKILL; inspect them with: ps -fp ${remaining[*]}" >&2
      exit 1
    fi
    sleep 1
  done
else
  echo "== No interrupted matrix processes found =="
fi

echo "== Restore inference Deployment from KVC burst overlay =="
NAMESPACE="$NAMESPACE" DEPLOYMENT="$DEPLOYMENT" BACKUP_FILE="$BACKUP_FILE" \
  bash "$SCRIPT_DIR/deploy_f14_kvc_burst_overlay.sh" restore

echo "== Verify recovered Deployment =="
kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" \
  --timeout="$ROLLOUT_TIMEOUT"

containers=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o jsonpath='{range .spec.template.spec.containers[*]}{.name}{"\n"}{end}')
if grep -Fxq kvc-burst-wrapper <<<"$containers"; then
  die "kvc-burst-wrapper remains in deployment/$DEPLOYMENT"
fi
env_names=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o jsonpath='{range .spec.template.spec.containers[*].env[*]}{.name}{"\n"}{end}')
if grep -q '^KVC_BURST_' <<<"$env_names"; then
  die "KVC_BURST environment remains in deployment/$DEPLOYMENT"
fi

kubectl -n "$NAMESPACE" get pods -l app="$DEPLOYMENT" -o wide
echo "containers=$(tr '\n' ',' <<<"$containers" | sed 's/,$//')"
echo "PAIREC_BRPC_WRAPPER_KVC_ENVIRONMENT_RECOVERED"
