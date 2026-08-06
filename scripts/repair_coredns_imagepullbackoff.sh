#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-kube-system}"
DEPLOYMENT="${DEPLOYMENT:-coredns}"
CONTAINER="${CONTAINER:-coredns}"
SERVICE="${SERVICE:-kube-dns}"
TARGET_IMAGE="${TARGET_IMAGE:-docker.io/coredns/coredns:1.8.3}"
TARGET_ARCH="${TARGET_ARCH:-arm64}"
SOURCE_HOST="${SOURCE_HOST:-root@192.168.100.11}"
PROBE_NAMESPACE="${PROBE_NAMESPACE:-pairec}"
PROBE_TARGET="${PROBE_TARGET:-deploy/pairec}"
INTERNAL_PROBE_NAMES="${INTERNAL_PROBE_NAMES:-inference-brpc-trtllm.pairec.svc.cluster.local milvus-standalone.pairec.svc.cluster.local}"
EXTERNAL_PROBE_NAME="${EXTERNAL_PROBE_NAME:-pypi.org}"
REQUIRE_EXTERNAL_DNS="${REQUIRE_EXTERNAL_DNS:-0}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-5m}"
PULL_TIMEOUT_SECONDS="${PULL_TIMEOUT_SECONDS:-180}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/coredns-repair-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/repair.log"
exec > >(tee "$LOG_FILE") 2>&1

TEMP_TAR=""
REMOTE_TAR=""

cleanup() {
  if [[ -n "$TEMP_TAR" ]]; then
    rm -f "$TEMP_TAR"
  fi
  if [[ -n "$REMOTE_TAR" && -n "$SOURCE_HOST" ]]; then
    ssh "$SOURCE_HOST" "rm -f '$REMOTE_TAR'" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

die() {
  local classification="$1"
  shift
  echo "ERROR: $*" >&2
  echo
  echo "== Summary =="
  echo "classification=$classification"
  echo "result=FAIL"
  echo "output_dir=$OUTPUT_DIR"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die MISSING_REQUIRED_COMMAND "missing command: $1"
}

ctr_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    ctr -n k8s.io "$@"
  else
    sudo ctr -n k8s.io "$@"
  fi
}

containerd_has_image() {
  ctr_cmd images list -q 2>/dev/null | grep -Fqx "$1"
}

first_local_coredns_image() {
  ctr_cmd images list -q 2>/dev/null |
    grep -Ei '(^|/)(coredns)(/coredns)?:v?1\.8\.3$' |
    head -1 || true
}

import_docker_image() {
  local image="$1"
  local arch
  arch="$(docker image inspect "$image" --format '{{.Architecture}}')"
  [[ "$arch" == "$TARGET_ARCH" ]] ||
    die COREDNS_IMAGE_ARCH_MISMATCH "image $image has architecture $arch, expected $TARGET_ARCH"

  TEMP_TAR="$(mktemp /tmp/coredns-image-XXXXXX.tar)"
  docker save -o "$TEMP_TAR" "$image"
  ctr_cmd images import "$TEMP_TAR"
}

import_remote_image() {
  local remote_ref
  remote_ref="$(ssh "$SOURCE_HOST" \
    "sudo ctr -n k8s.io images list -q 2>/dev/null | grep -Ei '(^|/)(coredns)(/coredns)?:v?1\\.8\\.3$' | head -1" || true)"
  [[ -n "$remote_ref" ]] || return 1

  echo "remote_source_image=$remote_ref"
  REMOTE_TAR="/tmp/coredns-1.8.3-${TARGET_ARCH}-$$.tar"
  ssh "$SOURCE_HOST" \
    "sudo ctr -n k8s.io images export '$REMOTE_TAR' '$remote_ref' && sudo chmod 0644 '$REMOTE_TAR'"
  TEMP_TAR="$(mktemp /tmp/coredns-remote-XXXXXX.tar)"
  scp "$SOURCE_HOST:$REMOTE_TAR" "$TEMP_TAR"
  ctr_cmd images import "$TEMP_TAR"

  if ! containerd_has_image "$TARGET_IMAGE"; then
    ctr_cmd images tag "$remote_ref" "$TARGET_IMAGE"
  fi
}

require_command kubectl
require_command ctr
require_command grep
require_command tee

echo "== CoreDNS repair configuration =="
echo "namespace=$NAMESPACE"
echo "deployment=$DEPLOYMENT"
echo "service=$SERVICE"
echo "target_image=$TARGET_IMAGE"
echo "target_arch=$TARGET_ARCH"
echo "source_host=$SOURCE_HOST"
echo "probe_target=$PROBE_NAMESPACE/$PROBE_TARGET"
echo "output_dir=$OUTPUT_DIR"

kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o yaml \
  >"$OUTPUT_DIR/deployment.before.yaml"
kubectl -n "$NAMESPACE" get service "$SERVICE" -o yaml \
  >"$OUTPUT_DIR/service.before.yaml"
kubectl -n "$NAMESPACE" get endpoints "$SERVICE" -o yaml \
  >"$OUTPUT_DIR/endpoints.before.yaml"

echo
echo "== Current state =="
kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o wide
kubectl -n "$NAMESPACE" get pods -l k8s-app=kube-dns -o wide
kubectl -n "$NAMESPACE" get endpoints "$SERVICE" -o wide
current_image="$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o "jsonpath={.spec.template.spec.containers[?(@.name=='$CONTAINER')].image}")"
current_policy="$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o "jsonpath={.spec.template.spec.containers[?(@.name=='$CONTAINER')].imagePullPolicy}")"
echo "current_image=$current_image"
echo "current_pull_policy=$current_policy"

ready_replicas="$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)"
ready_endpoints="$(kubectl -n "$NAMESPACE" get endpoints "$SERVICE" \
  -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || true)"

if [[ "${ready_replicas:-0}" != "0" && -n "$ready_endpoints" ]]; then
  echo "CoreDNS already has ready replicas and endpoints; image repair is not required."
else
  echo
  echo "== Ensure CoreDNS image in k8s.io containerd =="
  if containerd_has_image "$TARGET_IMAGE"; then
    echo "image_source=containerd_exact"
  else
    local_ref="$(first_local_coredns_image)"
    if [[ -n "$local_ref" ]]; then
      echo "image_source=containerd_alias"
      echo "local_source_image=$local_ref"
      ctr_cmd images tag "$local_ref" "$TARGET_IMAGE"
    elif command -v docker >/dev/null 2>&1 && docker image inspect "$TARGET_IMAGE" >/dev/null 2>&1; then
      echo "image_source=docker_local"
      import_docker_image "$TARGET_IMAGE"
    else
      pulled=0
      if command -v docker >/dev/null 2>&1; then
        echo "No local image found; trying a bounded Docker pull (${PULL_TIMEOUT_SECONDS}s)."
        if timeout "$PULL_TIMEOUT_SECONDS" docker pull --platform "linux/$TARGET_ARCH" "$TARGET_IMAGE"; then
          pulled=1
          echo "image_source=docker_pull"
          import_docker_image "$TARGET_IMAGE"
        else
          echo "Docker pull failed or timed out."
        fi
      fi

      if [[ "$pulled" == "0" && -n "$SOURCE_HOST" ]]; then
        echo "Trying local image transfer from $SOURCE_HOST."
        if import_remote_image; then
          echo "image_source=remote_containerd"
        fi
      fi
    fi
  fi

  containerd_has_image "$TARGET_IMAGE" ||
    die COREDNS_IMAGE_UNAVAILABLE \
      "target image is absent; import linux/$TARGET_ARCH $TARGET_IMAGE into k8s.io containerd and rerun"

  echo
  echo "== Patch CoreDNS deployment =="
  kubectl -n "$NAMESPACE" patch deployment "$DEPLOYMENT" --type=strategic -p \
    "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"$CONTAINER\",\"image\":\"$TARGET_IMAGE\",\"imagePullPolicy\":\"IfNotPresent\"}]}}}}"
  kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout="$WAIT_TIMEOUT" || {
    kubectl -n "$NAMESPACE" get pods -l k8s-app=kube-dns -o wide || true
    kubectl -n "$NAMESPACE" describe pods -l k8s-app=kube-dns \
      >"$OUTPUT_DIR/pods.describe.txt" 2>&1 || true
    kubectl -n "$NAMESPACE" logs -l k8s-app=kube-dns --tail=200 \
      >"$OUTPUT_DIR/coredns.log" 2>&1 || true
    die COREDNS_ROLLOUT_FAILED "CoreDNS did not become ready"
  }
fi

echo
echo "== Validate CoreDNS endpoints =="
kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o wide
kubectl -n "$NAMESPACE" get pods -l k8s-app=kube-dns -o wide
kubectl -n "$NAMESPACE" get endpoints "$SERVICE" -o wide
ready_endpoints="$(kubectl -n "$NAMESPACE" get endpoints "$SERVICE" \
  -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || true)"
[[ -n "$ready_endpoints" ]] || die COREDNS_ENDPOINTS_NOT_READY "kube-dns has no ready endpoint"

internal_status=SKIPPED
external_status=SKIPPED
if kubectl -n "$PROBE_NAMESPACE" get "$PROBE_TARGET" >/dev/null 2>&1; then
  echo
  echo "== Validate internal service DNS from $PROBE_NAMESPACE/$PROBE_TARGET =="
  internal_status=PASS
  for name in $INTERNAL_PROBE_NAMES; do
    if kubectl -n "$PROBE_NAMESPACE" exec "$PROBE_TARGET" -- nslookup "$name"; then
      echo "INTERNAL_DNS_OK=$name"
    else
      internal_status=FAIL
      echo "INTERNAL_DNS_FAILED=$name"
    fi
  done

  echo
  echo "== Validate external DNS (diagnostic by default) =="
  if kubectl -n "$PROBE_NAMESPACE" exec "$PROBE_TARGET" -- nslookup "$EXTERNAL_PROBE_NAME"; then
    external_status=PASS
    echo "EXTERNAL_DNS_OK=$EXTERNAL_PROBE_NAME"
  else
    external_status=FAIL
    echo "EXTERNAL_DNS_FAILED=$EXTERNAL_PROBE_NAME"
  fi
fi

[[ "$internal_status" != "FAIL" ]] ||
  die COREDNS_INTERNAL_LOOKUP_FAILED "CoreDNS is ready but internal service lookup failed"
if [[ "$REQUIRE_EXTERNAL_DNS" == "1" && "$external_status" != "PASS" ]]; then
  die COREDNS_EXTERNAL_LOOKUP_FAILED "external DNS is required but lookup failed"
fi

classification=COREDNS_INTERNAL_DNS_RESTORED
if [[ "$external_status" == "FAIL" ]]; then
  classification=COREDNS_INTERNAL_OK_EXTERNAL_DNS_FAILED
fi

echo
echo "== Summary =="
echo "classification=$classification"
echo "result=PASS"
echo "target_image=$TARGET_IMAGE"
echo "ready_endpoints=$ready_endpoints"
echo "internal_dns_status=$internal_status"
echo "external_dns_status=$external_status"
echo "deployment_backup=$OUTPUT_DIR/deployment.before.yaml"
echo "output_dir=$OUTPUT_DIR"
echo "COREDNS_IMAGEPULLBACKOFF_REPAIR_COMPLETE"
