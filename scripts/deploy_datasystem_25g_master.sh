#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAMESPACE="${NAMESPACE:-pairec}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/k8s/deployment-datasystem-25g-master.yaml}"
DEPLOYMENT="${DEPLOYMENT:-datasystem-25g-master}"
MASTER_25G_IP="${MASTER_25G_IP:-192.168.100.12}"
DS_ENDPOINT="${DS_ENDPOINT:-${MASTER_25G_IP}:18482}"
ETCD_ENDPOINT="${ETCD_ENDPOINT:-141.61.91.189:12379}"
KVC_LOAD_HOST="${KVC_LOAD_HOST:-root@141.61.91.188}"
KVC_DSBENCH_CPP="${KVC_DSBENCH_CPP:-/home/zcx/bin/dsbench-v081-sustained}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"
ALLOW_REPLACE_UNKNOWN_LISTENER="${ALLOW_REPLACE_UNKNOWN_LISTENER:-0}"

log() {
  printf '\n== %s ==\n' "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f "$MANIFEST" ] || die "manifest does not exist: ${MANIFEST}"
command -v kubectl >/dev/null 2>&1 || die "kubectl is required"
command -v ip >/dev/null 2>&1 || die "ip is required"
case "$ALLOW_REPLACE_UNKNOWN_LISTENER" in
  0|1) ;;
  *) die "ALLOW_REPLACE_UNKNOWN_LISTENER must be 0 or 1" ;;
esac

DS_HOST="${DS_ENDPOINT%:*}"
DS_PORT="${DS_ENDPOINT##*:}"
ETCD_HOST="${ETCD_ENDPOINT%:*}"
ETCD_PORT="${ETCD_ENDPOINT##*:}"
[ "$DS_HOST" = "$MASTER_25G_IP" ] \
  || die "DS_ENDPOINT host must equal MASTER_25G_IP"
[[ "$DS_PORT" =~ ^[1-9][0-9]*$ ]] || die "invalid DataSystem port"
[[ "$ETCD_PORT" =~ ^[1-9][0-9]*$ ]] || die "invalid etcd port"
[ "$MASTER_25G_IP" = "192.168.100.12" ] \
  || die "strict 25G experiment requires MASTER_25G_IP=192.168.100.12"
[ "$DS_ENDPOINT" = "192.168.100.12:18482" ] \
  || die "strict 25G experiment requires DS_ENDPOINT=192.168.100.12:18482"
[ "$ETCD_ENDPOINT" = "141.61.91.189:12379" ] \
  || die "strict 25G experiment requires ETCD_ENDPOINT=141.61.91.189:12379"

log "25G host preflight"
ip -o address show | tee /tmp/pairec-datasystem-25g-addresses.log
if ! ip -o address show | grep -Fq " ${MASTER_25G_IP}/"; then
  echo "Run the non-mutating link diagnosis first:" >&2
  echo "  APPLY=0 bash scripts/restore_strict_25g_link.sh" >&2
  die "master does not own 25G address ${MASTER_25G_IP}"
fi

if ! timeout 3 bash -c "</dev/tcp/${ETCD_HOST}/${ETCD_PORT}" 2>/dev/null; then
  die "etcd is not reachable: ${ETCD_ENDPOINT}"
fi
echo "ETCD_TCP_OK endpoint=${ETCD_ENDPOINT}"

EXISTING_DEPLOYMENT=0
if kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" >/dev/null 2>&1; then
  EXISTING_DEPLOYMENT=1
fi

if timeout 2 bash -c "</dev/tcp/${DS_HOST}/${DS_PORT}" 2>/dev/null; then
  if [ "$EXISTING_DEPLOYMENT" -ne 1 ] && [ "$ALLOW_REPLACE_UNKNOWN_LISTENER" -ne 1 ]; then
    ss -lntp 2>/dev/null | grep -E ":${DS_PORT}([[:space:]]|$)" || true
    die "${DS_ENDPOINT} already has an unknown listener; refusing to replace it"
  fi
  echo "EXISTING_DS_LISTENER endpoint=${DS_ENDPOINT} managed_deployment=${EXISTING_DEPLOYMENT}"
else
  echo "DS_PORT_AVAILABLE endpoint=${DS_ENDPOINT}"
fi

log "Preserved DataSystem pool"
kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/part-of=datasystem-pool -o wide

log "Deploy isolated 25G DataSystem worker"
kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" rollout status "deployment/${DEPLOYMENT}" \
  --timeout="$ROLLOUT_TIMEOUT"
kubectl -n "$NAMESPACE" get pods -l "app=${DEPLOYMENT}" -o wide

POD="$(
  kubectl -n "$NAMESPACE" get pods -l "app=${DEPLOYMENT}" \
    -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' \
    | head -1
)"
[ -n "$POD" ] || die "no running ${DEPLOYMENT} pod was found"

log "Worker process and listening endpoint"
kubectl -n "$NAMESPACE" exec "$POD" -- bash -lc \
  'pgrep -af datasystem_worker; grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/1/status; df -h /dev/shm'

log "188 to 25G Worker TCP"
ssh "$KVC_LOAD_HOST" \
  "timeout 3 bash -c '</dev/tcp/${DS_HOST}/${DS_PORT}'" \
  || die "${KVC_LOAD_HOST} cannot reach ${DS_ENDPOINT}"
echo "DATASYSTEM_25G_TCP_OK load_host=${KVC_LOAD_HOST} endpoint=${DS_ENDPOINT}"

log "1KB DataSystem RPC smoke"
KVC_LOAD_HOST="$KVC_LOAD_HOST" \
KVC_DS_ENDPOINT="$DS_ENDPOINT" \
KVC_DSBENCH_CPP="$KVC_DSBENCH_CPP" \
OUT_DIR="/tmp/datasystem-worker-rpc/deploy-25g-$(date +%Y%m%d%H%M%S)" \
  bash "${SCRIPT_DIR}/diagnose_datasystem_worker_rpc.sh"

log "Inference endpoint contract"
INFERENCE_DS_ENV="$(
  kubectl -n "$NAMESPACE" get deployment inference-brpc-trtllm \
    -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}' \
    | grep -E '^DATASYSTEM_(HOST|PORT)='
)"
printf '%s\n' "$INFERENCE_DS_ENV"
grep -qx "DATASYSTEM_HOST=${DS_HOST}" <<<"$INFERENCE_DS_ENV" \
  || die "inference DATASYSTEM_HOST does not match ${DS_HOST}"
grep -qx "DATASYSTEM_PORT=${DS_PORT}" <<<"$INFERENCE_DS_ENV" \
  || die "inference DATASYSTEM_PORT does not match ${DS_PORT}"

echo "DATASYSTEM_25G_MASTER_DEPLOYMENT_OK"
echo "container_pod=${POD}"
echo "endpoint=${DS_ENDPOINT}"
echo "next=bash scripts/diagnose_datasystem_sustained_get_lifecycle.sh"
