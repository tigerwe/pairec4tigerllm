#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/zcx-pairec-brpc-sdk:v1}"
BASE_IMAGE="${BASE_IMAGE:-${2:-zcx-pairec-image:v1.1}}"
BRPC_REPO="${BRPC_REPO:-https://github.com/apache/brpc.git}"
BRPC_REF="${BRPC_REF:-master}"

TMP_DIR="$(mktemp -d /tmp/pairec-brpc-sdk.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cat > "$TMP_DIR/Dockerfile" <<'EOF'
ARG BASE_IMAGE=zcx-pairec-image:v1.1
ARG BRPC_REPO=https://github.com/apache/brpc.git
ARG BRPC_REF=master

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

RUN if command -v dnf >/dev/null 2>&1; then \
      dnf install -y git gcc gcc-c++ make cmake openssl-devel gflags-devel protobuf-devel protobuf-compiler leveldb-devel zlib-devel && dnf clean all; \
    elif command -v yum >/dev/null 2>&1; then \
      yum install -y git gcc gcc-c++ make cmake openssl-devel gflags-devel protobuf-devel protobuf-compiler leveldb-devel zlib-devel && yum clean all; \
    elif command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends git g++ make cmake libssl-dev libgflags-dev libprotobuf-dev libprotoc-dev protobuf-compiler libleveldb-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*; \
    else \
      echo "No supported package manager found" >&2; exit 1; \
    fi

RUN git clone --depth=1 --branch "${BRPC_REF}" "${BRPC_REPO}" /tmp/brpc

RUN cmake -S /tmp/brpc -B /tmp/brpc/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_BRPC_TOOLS=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      -DBUILD_FUZZ_TESTS=OFF \
      -DDOWNLOAD_GTEST=OFF \
      -DWITH_DEBUG_SYMBOLS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
 && cmake --build /tmp/brpc/build -j"$(nproc)" \
 && cmake --install /tmp/brpc/build \
 && ldconfig \
 && test -f /usr/local/include/brpc/server.h \
 && find /usr/local -name 'libbrpc*' -print
EOF

echo "Building brpc SDK image"
echo "  image:      $IMAGE"
echo "  base image: $BASE_IMAGE"
echo "  brpc repo:  $BRPC_REPO"
echo "  brpc ref:   $BRPC_REF"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "BRPC_REPO=$BRPC_REPO" \
  --build-arg "BRPC_REF=$BRPC_REF" \
  -f "$TMP_DIR/Dockerfile" \
  -t "$IMAGE" \
  "$TMP_DIR"

echo ""
echo "Built $IMAGE"
echo "Verify with:"
echo "  docker run --rm $IMAGE bash -lc 'ls -l /usr/local/include/brpc/server.h; find /usr/local -name \"libbrpc*\" -print; protoc --version'"
