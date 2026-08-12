#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-trtllm-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zcx/pairec-f19-historical-runtime}"
CTR="${CTR:-ctr}"
IMAGE_TAR="${IMAGE_TAR:-/home/zcx/pairec-brpc-inference-k8s-arm64-trtllm-v1.tar}"
EXPECTED_GATEWAY_SHA256="${EXPECTED_GATEWAY_SHA256:-acba014e342030a57e1fba51fd691b1fbb7ffd3735488f03b28e08365b00dc43}"
EXPECTED_TRTLLM_SHA256="${EXPECTED_TRTLLM_SHA256:-e0452812c00a56ae9a0b5817a5c0ca6b1a2e1b8e33dc63fe22c31e2e834010c4}"
MOUNT_DIR=""

die() { echo "ERROR: $*" >&2; exit 1; }

cleanup() {
  if [[ -n "$MOUNT_DIR" && -d "$MOUNT_DIR" ]]; then
    sudo "$CTR" -n k8s.io images unmount "$MOUNT_DIR" >/dev/null 2>&1 || true
    rmdir "$MOUNT_DIR" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for command in sudo "$CTR" install sha256sum strings mktemp grep awk find readlink; do
  command -v "$command" >/dev/null || die "missing command: $command"
done

image_exists() {
  sudo "$CTR" -n k8s.io images list -q 2>/dev/null | grep -Fqx "$IMAGE"
}

if ! image_exists; then
  [[ -f "$IMAGE_TAR" ]] \
    || die "historical image is absent and archive does not exist: image=$IMAGE archive=$IMAGE_TAR"
  echo "== Import historical inference image archive =="
  echo "archive=$IMAGE_TAR"
  sudo "$CTR" -n k8s.io images import "$IMAGE_TAR"
  image_exists \
    || die "archive import did not create expected image tag: $IMAGE"
else
  echo "historical_image_source=containerd"
fi

MOUNT_DIR="$(mktemp -d /tmp/f19-historical-image.XXXXXX)"
echo "== Mount historical inference image =="
echo "image=$IMAGE"
echo "mount_dir=$MOUNT_DIR"
sudo "$CTR" -n k8s.io images mount "$IMAGE" "$MOUNT_DIR" >/dev/null

gateway="$MOUNT_DIR/opt/pairec-brpc/bin/brpc_inference_server"
[[ -x "$gateway" ]] || die "historical gateway is missing: $gateway"

echo "== Locate historical TensorRT-LLM library by SHA =="
mapfile -t trt_candidates < <(
  find "$MOUNT_DIR" -xdev -maxdepth 10 \
    \( -type f -o -type l \) -name libtensorrt_llm.so -print
)
resolved_candidates=()
for candidate in "${trt_candidates[@]}"; do
  resolved="$candidate"
  if [[ -L "$candidate" ]]; then
    target="$(readlink "$candidate")"
    if [[ "$target" == /* ]]; then
      resolved="$MOUNT_DIR$target"
    else
      resolved="$(readlink -f "$(dirname "$candidate")/$target" 2>/dev/null || true)"
    fi
  fi
  [[ -f "$resolved" ]] || continue
  duplicate=0
  for existing in "${resolved_candidates[@]}"; do
    if [[ "$existing" == "$resolved" ]]; then duplicate=1; break; fi
  done
  (( duplicate == 1 )) || resolved_candidates+=("$resolved")
done

trt_library=""
for candidate in "${resolved_candidates[@]}"; do
  candidate_hash="$(sha256sum "$candidate" | awk '{print $1}')"
  echo "candidate=${candidate#"$MOUNT_DIR"} sha256=$candidate_hash"
  if [[ -n "$EXPECTED_TRTLLM_SHA256" && "$candidate_hash" == "$EXPECTED_TRTLLM_SHA256" ]]; then
    trt_library="$candidate"
  fi
done
if [[ -z "$trt_library" && -z "$EXPECTED_TRTLLM_SHA256" \
    && "${#resolved_candidates[@]}" -eq 1 ]]; then
  trt_library="${resolved_candidates[0]}"
fi
[[ -n "$trt_library" ]] || die \
  "no unique historical TRT library matched expected SHA under image rootfs"

mkdir -p "$OUTPUT_DIR/bin" "$OUTPUT_DIR/lib"
install -m 0755 "$gateway" "$OUTPUT_DIR/bin/brpc_inference_server"
install -m 0755 "$trt_library" "$OUTPUT_DIR/lib/libtensorrt_llm.so"

echo "== Historical runtime evidence =="
hashes="$(sha256sum \
  "$OUTPUT_DIR/bin/brpc_inference_server" \
  "$OUTPUT_DIR/lib/libtensorrt_llm.so")"
echo "$hashes"
gateway_hash="$(awk 'NR==1 {print $1}' <<<"$hashes")"
trtllm_hash="$(awk 'NR==2 {print $1}' <<<"$hashes")"
[[ -z "$EXPECTED_GATEWAY_SHA256" || "$gateway_hash" == "$EXPECTED_GATEWAY_SHA256" ]] \
  || die "historical gateway SHA mismatch: actual=$gateway_hash expected=$EXPECTED_GATEWAY_SHA256"
[[ -z "$EXPECTED_TRTLLM_SHA256" || "$trtllm_hash" == "$EXPECTED_TRTLLM_SHA256" ]] \
  || die "historical TRT SHA mismatch: actual=$trtllm_hash expected=$EXPECTED_TRTLLM_SHA256"
if strings "$OUTPUT_DIR/bin/brpc_inference_server" | grep -Fq output_token_count; then
  echo "output_token_trace=present"
else
  echo "output_token_trace=absent"
fi
echo "F19_HISTORICAL_RUNTIME_EXTRACT_OK output_dir=$OUTPUT_DIR"
