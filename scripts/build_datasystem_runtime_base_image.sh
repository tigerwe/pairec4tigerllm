#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-3d25ebe028d6}"
IMAGE="${2:-docker.io/library/zcx-pairec-ds-runtime:v1}"
BASE_IMAGE="${3:-zcx-pairec-image:v1.1}"

TMP_DIR="$(mktemp -d /tmp/pairec-ds-runtime-base.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "Source container: $CONTAINER"
echo "Base image:       $BASE_IMAGE"
echo "Output image:     $IMAGE"
echo ""
echo "Container mounts:"
docker inspect "$CONTAINER" \
  --format '{{range .Mounts}}{{println .Destination "->" .Source}}{{end}}'

docker exec "$CONTAINER" env -u LD_PRELOAD bash -lc \
  'test -f /TensorRT-LLM/tensorrt_llm/__init__.py'

echo ""
echo "Copying patched /TensorRT-LLM from the source container..."
docker cp "$CONTAINER:/TensorRT-LLM" "$TMP_DIR/TensorRT-LLM"

cat > "$TMP_DIR/Dockerfile" <<'EOF'
ARG BASE_IMAGE=zcx-pairec-image:v1.1
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

RUN rm -rf /home/TensorRT-LLM /TensorRT-LLM
COPY TensorRT-LLM /home/TensorRT-LLM
RUN ln -s /home/TensorRT-LLM /TensorRT-LLM

ENV PYTHONPATH=/app:/home/TensorRT-LLM:/TensorRT-LLM:/home/TensorRT-LLM/3rdparty/cutlass/python:/TensorRT-LLM/3rdparty/cutlass/python
EOF

echo ""
echo "Building $IMAGE ..."
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$TMP_DIR/Dockerfile" \
  -t "$IMAGE" \
  "$TMP_DIR"

echo ""
echo "Built $IMAGE"
echo "Next:"
echo "  bash scripts/build_inference_runtime_image.sh \\"
echo "    docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1 \\"
echo "    /home/zcx/pairec-inference-k8s-arm64-ds-runtime-v1.tar \\"
echo "    $IMAGE"
