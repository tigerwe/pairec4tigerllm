#!/usr/bin/env bash
set -u -o pipefail

OUTPUT_BASE=${OUTPUT_BASE:-/dev/shm}
if [[ ! -d "$OUTPUT_BASE" || ! -w "$OUTPUT_BASE" ]]; then
  OUTPUT_BASE=/tmp
fi
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_BASE/k8s-control-plane-diagnostic/$(date +%Y%m%d-%H%M%S)}
KUBELET_LOG=${KUBELET_LOG:-/root/kubernetes/log/kubelet/kubelet.log}

mkdir -p "$OUTPUT_DIR" || {
  echo "ERROR: cannot create diagnostic output directory: $OUTPUT_DIR" >&2
  exit 1
}

if (( EUID == 0 )); then
  SUDO=()
else
  SUDO=(sudo -n)
fi

have() { command -v "$1" >/dev/null 2>&1; }

capture() {
  local name=$1
  shift
  {
    echo "command=$*"
    echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
    "$@"
    echo "exit_code=$?"
  } >"$OUTPUT_DIR/$name" 2>&1 || true
}

echo "== Kubernetes control-plane and root-disk diagnosis =="
echo "output_dir=$OUTPUT_DIR"

capture df.txt df -hT / /tmp /var/log /root/kubernetes/log
capture df-inodes.txt df -ih / /tmp /var/log /root/kubernetes/log
capture mounts.txt findmnt -T /
capture memory.txt free -h
capture kubelet-status.txt "${SUDO[@]}" systemctl status kubelet -l --no-pager
capture kubelet-properties.txt "${SUDO[@]}" systemctl show kubelet \
  -p ActiveState -p SubState -p Result -p ExecMainCode -p ExecMainStatus \
  -p NRestarts
capture kubelet-journal.txt "${SUDO[@]}" journalctl -u kubelet \
  --since "30 minutes ago" --no-pager -n 500

if [[ -f "$KUBELET_LOG" ]]; then
  capture kubelet-file-log.txt "${SUDO[@]}" tail -n 500 "$KUBELET_LOG"
else
  printf 'missing=%s\n' "$KUBELET_LOG" >"$OUTPUT_DIR/kubelet-file-log.txt"
fi

capture root-top-level-bytes.txt "${SUDO[@]}" du -x -B1 -d1 /
capture root-top-level-human.txt "${SUDO[@]}" du -x -h -d1 /
capture root-log-top-level.txt "${SUDO[@]}" du -x -h -d2 /root/kubernetes/log
capture var-log-top-level.txt "${SUDO[@]}" du -x -h -d2 /var/log
capture tmp-top-level.txt "${SUDO[@]}" du -x -h -d1 /tmp

if have findmnt && have lsof; then
  root_device=$(findmnt -n -o MAJ:MIN / 2>/dev/null | tr ':' ',')
  {
    echo "root_device=$root_device"
    "${SUDO[@]}" lsof +L1 -nP 2>/dev/null |
      awk -v device="$root_device" 'NR == 1 || $5 == device'
  } >"$OUTPUT_DIR/root-deleted-open-files.txt" 2>&1 || true
else
  echo "lsof or findmnt unavailable" >"$OUTPUT_DIR/root-deleted-open-files.txt"
fi

if have lsof && have findmnt; then
  root_device=$(findmnt -n -o MAJ:MIN / 2>/dev/null | tr ':' ',')
  deleted_open_bytes=$("${SUDO[@]}" lsof +L1 -nP 2>/dev/null |
    awk -v device="$root_device" '
      $5 == device && !seen[$8]++ && $6 ~ /^[0-9]+$/ { total += $6 }
      END { printf "%.0f", total + 0 }
    ')
else
  deleted_open_bytes=0
fi

if have crictl; then
  capture control-plane-containers.txt "${SUDO[@]}" timeout 15s crictl ps -a
else
  echo "crictl unavailable" >"$OUTPUT_DIR/control-plane-containers.txt"
fi

if have curl; then
  api_http_code=$(curl -sk --max-time 3 -o "$OUTPUT_DIR/api-livez.txt" \
    -w '%{http_code}' https://127.0.0.1:6443/livez 2>/dev/null || true)
else
  api_http_code=000
  echo "curl unavailable" >"$OUTPUT_DIR/api-livez.txt"
fi

if have kubectl; then
  capture kubectl-readyz.txt timeout 10s kubectl get --raw=/readyz?verbose
  capture kubectl-nodes.txt timeout 10s kubectl get nodes -o wide
else
  echo "kubectl unavailable" >"$OUTPUT_DIR/kubectl-readyz.txt"
  echo "kubectl unavailable" >"$OUTPUT_DIR/kubectl-nodes.txt"
fi

capture static-pod-manifests.txt "${SUDO[@]}" find \
  /root/kubernetes/etc/kubernetes -maxdepth 3 -type f -name '*.yaml' -ls

capture largest-root-files.txt "${SUDO[@]}" find / -xdev -type f -size +100M \
  -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n'

root_used_pct=$(df -P / | awk 'NR == 2 { gsub(/%/, "", $5); print $5 + 0 }')
root_available_kb=$(df -Pk / | awk 'NR == 2 { print $4 + 0 }')
kubelet_state=$(systemctl is-active kubelet 2>/dev/null || true)
control_plane_count=0
if [[ -f "$OUTPUT_DIR/control-plane-containers.txt" ]]; then
  control_plane_count=$(grep -Ec 'kube-apiserver|kube-controller-manager|kube-scheduler' \
    "$OUTPUT_DIR/control-plane-containers.txt" || true)
fi

classification=K8S_CONTROL_PLANE_DIAGNOSTIC_OK
next_action="control plane and root disk are healthy"
if (( root_used_pct >= 98 )) && [[ "$kubelet_state" != "active" ]] && [[ "$api_http_code" != "200" ]]; then
  classification=ROOT_DISK_FULL_KUBELET_CONTROL_PLANE_FAILURE
  next_action="free 5-10GiB from confirmed logs or stale artifacts, then restart kubelet"
elif (( root_used_pct >= 98 )); then
  classification=ROOT_DISK_CRITICAL
  next_action="free 5-10GiB before running another benchmark"
elif [[ "$kubelet_state" != "active" ]]; then
  classification=KUBELET_FAILURE
  next_action="inspect kubelet-file-log.txt and kubelet-journal.txt before restarting kubelet"
elif [[ "$api_http_code" != "200" ]]; then
  classification=KUBE_API_UNAVAILABLE
  next_action="inspect control-plane-containers.txt and static-pod manifests"
fi

{
  echo "classification=$classification"
  echo "root_used_pct=$root_used_pct"
  echo "root_available_kb=$root_available_kb"
  echo "root_device=${root_device:-unknown}"
  echo "root_deleted_open_bytes=$deleted_open_bytes"
  echo "kubelet_state=${kubelet_state:-unknown}"
  echo "api_livez_http_code=${api_http_code:-000}"
  echo "control_plane_container_matches=$control_plane_count"
  echo "datasystem_memfd_note=memfd device 0,1 is shared memory and is excluded from root-disk deleted-file totals"
  echo "next_action=$next_action"
} | tee "$OUTPUT_DIR/summary.txt"

echo ""
echo "== Largest top-level root directories =="
sort -n "$OUTPUT_DIR/root-top-level-bytes.txt" 2>/dev/null | tail -15 || true
echo ""
echo "== Root-filesystem deleted-open files =="
tail -20 "$OUTPUT_DIR/root-deleted-open-files.txt" 2>/dev/null || true
echo ""
echo "$classification"
