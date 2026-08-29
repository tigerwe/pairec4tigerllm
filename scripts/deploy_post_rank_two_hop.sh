#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
BUILD_IMAGE="${BUILD_IMAGE:-docker.io/library/pairec-brpc-inference:post-rank-two-hop-20260829}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
BINARY_PATH="${BINARY_PATH:-/home/zcx/bin/brpc_post_rank_hop}"
HOP1_MANIFEST="${HOP1_MANIFEST:-k8s/deployment-post-rank-hop1.yaml}"
HOP2_MANIFEST="${HOP2_MANIFEST:-k8s/deployment-post-rank-hop2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/post-rank-two-hop-deploy-$(date +%Y%m%d-%H%M%S)}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"

die() { echo "ERROR: $*" >&2; exit 1; }
for command in kubectl docker; do command -v "$command" >/dev/null || die "missing command: $command"; done
docker image inspect "$BUILD_IMAGE" >/dev/null || die "missing build image: $BUILD_IMAGE"
mkdir -p "$OUTPUT_DIR"

echo "== Extract post-rank Hop binary from build image =="
container="$(docker create "$BUILD_IMAGE")"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
mkdir -p "$(dirname "$BINARY_PATH")"
docker cp "$container:/opt/pairec-brpc/bin/brpc_post_rank_hop" "$BINARY_PATH.part"
chmod 0755 "$BINARY_PATH.part"
mv -f "$BINARY_PATH.part" "$BINARY_PATH"
docker rm -f "$container" >/dev/null
trap - EXIT

render() {
  python3 - "$1" "$2" "$IMAGE" <<'PY'
import pathlib, sys
source, target, image = sys.argv[1:]
text = pathlib.Path(source).read_text().replace("__IMAGE__", image)
assert "__" not in text
pathlib.Path(target).write_text(text)
PY
}

IMAGE="$RUNTIME_IMAGE"
render "$HOP2_MANIFEST" "$OUTPUT_DIR/hop2.yaml"
render "$HOP1_MANIFEST" "$OUTPUT_DIR/hop1.yaml"

echo "== Deploy post-rank Hop-2 first =="
kubectl apply -f "$OUTPUT_DIR/hop2.yaml"
kubectl -n "$NAMESPACE" rollout restart deployment/post-rank-hop2
kubectl -n "$NAMESPACE" rollout status deployment/post-rank-hop2 --timeout="$ROLLOUT_TIMEOUT"

echo "== Deploy post-rank Hop-1 and preconnect c1000 to Hop-2 =="
kubectl apply -f "$OUTPUT_DIR/hop1.yaml"
kubectl -n "$NAMESPACE" rollout restart deployment/post-rank-hop1
kubectl -n "$NAMESPACE" rollout status deployment/post-rank-hop1 --timeout="$ROLLOUT_TIMEOUT"

HOP1_POD="$(kubectl -n "$NAMESPACE" get pod -l app=post-rank-hop1 -o jsonpath='{.items[0].metadata.name}')"
HOP2_POD="$(kubectl -n "$NAMESPACE" get pod -l app=post-rank-hop2 -o jsonpath='{.items[0].metadata.name}')"
kubectl -n "$NAMESPACE" logs "$HOP1_POD" -c post-rank-hop1 --tail=200 >"$OUTPUT_DIR/hop1.log"
kubectl -n "$NAMESPACE" logs "$HOP2_POD" -c post-rank-hop2 --tail=50 >"$OUTPUT_DIR/hop2.log"
grep -Fq '"event":"pairec_post_rank_hop2_brpc_burst_ready"' "$OUTPUT_DIR/hop1.log" \
  || die "Hop-1 did not prove 1000 preconnected Hop-2 sessions"
grep -Fq '"connected_sessions":1000' "$OUTPUT_DIR/hop1.log" \
  || die "Hop-1 connected session count is not 1000"
grep -Fq 'post-rank hop2 pressure listening on 0.0.0.0:18313' "$OUTPUT_DIR/hop2.log" \
  || die "Hop-2 pressure listener is not ready on 18313"

HOST_BINARY_SHA256="$(sha256sum "$BINARY_PATH" | awk '{print $1}')"
for tuple in "$HOP1_POD:post-rank-hop1" "$HOP2_POD:post-rank-hop2"; do
  IFS=: read -r pod container <<<"$tuple"
  POD_BINARY_SHA256="$(kubectl -n "$NAMESPACE" exec "$pod" -c "$container" -- \
    sha256sum /opt/pairec-brpc-mounted/brpc_post_rank_hop | awk '{print $1}')"
  [[ "$POD_BINARY_SHA256" = "$HOST_BINARY_SHA256" ]] \
    || die "running post-rank binary mismatch pod=$pod host_sha=$HOST_BINARY_SHA256 pod_sha=$POD_BINARY_SHA256"
done
printf 'binary_sha256=%s\n' "$HOST_BINARY_SHA256" >"$OUTPUT_DIR/binary-sha256.txt"

echo "PAIREC_POST_RANK_TWO_HOP_READY hop1=192.168.100.12:18311 hop2_business=192.168.100.12:18312 hop2_pressure=192.168.100.12:18313"
echo "binary_sha256=$HOST_BINARY_SHA256"
echo "output_dir=$OUTPUT_DIR"
