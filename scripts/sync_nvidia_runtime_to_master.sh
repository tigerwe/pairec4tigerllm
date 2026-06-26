#!/usr/bin/env bash
set -euo pipefail

SOURCE_NODE="${SOURCE_NODE:-141.61.91.188}"
MASTER_NODE_NAME="${MASTER_NODE_NAME:-master}"
SOURCE_IMPORT="${SOURCE_IMPORT:-/etc/containerd/conf.d/99-nvidia.toml}"
TARGET_CONFIG="${TARGET_CONFIG:-/etc/containerd/config.toml}"
TARGET_IMPORT="${TARGET_IMPORT:-/etc/containerd/conf.d/99-nvidia.toml}"
RESTART_SERVICES="${RESTART_SERVICES:-1}"
DELETE_NVIDIA_PLUGIN_PODS="${DELETE_NVIDIA_PLUGIN_PODS:-1}"
LABEL_GPU_NODE="${LABEL_GPU_NODE:-1}"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

echo "Sync NVIDIA containerd runtime config to local master node"
echo "  source node:   ${SOURCE_NODE}"
echo "  source import: ${SOURCE_IMPORT}"
echo "  target config: ${TARGET_CONFIG}"
echo "  target import: ${TARGET_IMPORT}"
echo

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh is required" >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "ERROR: sudo is required" >&2
  exit 1
fi

if [ ! -x /usr/bin/nvidia-container-runtime ]; then
  echo "ERROR: /usr/bin/nvidia-container-runtime is missing or not executable on this node" >&2
  exit 1
fi

echo "== Fetch ${SOURCE_IMPORT} from ${SOURCE_NODE} =="
ssh "$SOURCE_NODE" "sudo cat '${SOURCE_IMPORT}'" > "${tmpdir}/99-nvidia.toml"

if ! grep -q "nvidia-container-runtime" "${tmpdir}/99-nvidia.toml"; then
  echo "ERROR: fetched ${SOURCE_IMPORT} does not look like an NVIDIA runtime config" >&2
  exit 1
fi

echo "== Install ${TARGET_IMPORT} =="
sudo mkdir -p "$(dirname "$TARGET_IMPORT")"
sudo install -m 0644 "${tmpdir}/99-nvidia.toml" "$TARGET_IMPORT"

if [ ! -f "$TARGET_CONFIG" ]; then
  echo "ERROR: target containerd config not found: ${TARGET_CONFIG}" >&2
  exit 1
fi

backup="${TARGET_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
echo "== Backup ${TARGET_CONFIG} -> ${backup} =="
sudo cp "$TARGET_CONFIG" "$backup"

echo "== Ensure containerd imports ${TARGET_IMPORT} =="
if sudo grep -qF "$TARGET_IMPORT" "$TARGET_CONFIG"; then
  echo "import already present"
else
  work="${tmpdir}/config.toml"
  if sudo grep -qE '^[[:space:]]*imports[[:space:]]*=' "$TARGET_CONFIG"; then
    sudo awk -v import_path="$TARGET_IMPORT" '
      BEGIN { updated = 0 }
      !updated && $0 ~ /^[[:space:]]*imports[[:space:]]*=/ {
        line = $0
        if (line ~ /\[[[:space:]]*\]/) {
          sub(/\[[[:space:]]*\]/, "[\"" import_path "\"]", line)
        } else {
          sub(/[[:space:]]*\][[:space:]]*$/, ", \"" import_path "\"]", line)
        }
        print line
        updated = 1
        next
      }
      { print }
    ' "$TARGET_CONFIG" > "$work"
  else
    printf 'imports = ["%s"]\n' "$TARGET_IMPORT" > "$work"
    sudo cat "$TARGET_CONFIG" >> "$work"
  fi
  sudo install -m 0644 "$work" "$TARGET_CONFIG"
fi

echo
echo "== Local containerd config before restart =="
sudo containerd config dump | grep -n "nvidia" -C 8 || true

if [ "$RESTART_SERVICES" = "1" ]; then
  echo
  echo "== Restart containerd and kubelet =="
  sudo systemctl restart containerd
  sudo systemctl restart kubelet
else
  echo
  echo "Skipped service restart because RESTART_SERVICES=${RESTART_SERVICES}"
fi

echo
echo "== Local CRI runtime check =="
sudo containerd config dump | grep -n "nvidia" -C 8 || true
sudo crictl info | grep -n "nvidia" -C 8 || true

if command -v kubectl >/dev/null 2>&1; then
  if [ "$LABEL_GPU_NODE" = "1" ]; then
    echo
    echo "== Ensure node label pairec/gpu=true on ${MASTER_NODE_NAME} =="
    kubectl label node "$MASTER_NODE_NAME" pairec/gpu=true --overwrite
  fi

  if [ "$DELETE_NVIDIA_PLUGIN_PODS" = "1" ]; then
    echo
    echo "== Recreate NVIDIA device plugin pods =="
    mapfile -t nvidia_pods < <(kubectl -n kube-system get pods -o name | grep 'nvidia-device-plugin-daemonset' || true)
    if [ "${#nvidia_pods[@]}" -gt 0 ]; then
      kubectl -n kube-system delete "${nvidia_pods[@]}" --force --grace-period=0
    else
      echo "no existing nvidia-device-plugin-daemonset pods found"
    fi
  fi

  echo
  echo "== NVIDIA device plugin pods =="
  kubectl -n kube-system get pods -o wide | grep -i nvidia || true

  echo
  echo "== Master GPU allocatable check =="
  kubectl describe node "$MASTER_NODE_NAME" | grep -A8 -E "Capacity:|Allocatable:|nvidia.com/gpu" || true
else
  echo
  echo "kubectl not found; skipped Kubernetes checks"
fi

echo
echo "Next expected success condition:"
echo "  kubectl describe node ${MASTER_NODE_NAME} | grep -A8 -E 'Capacity:|Allocatable:|nvidia.com/gpu'"
echo "should show:"
echo "  nvidia.com/gpu: 1"
