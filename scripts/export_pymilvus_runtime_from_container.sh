#!/usr/bin/env bash
set -euo pipefail

SOURCE_CONTAINER="${SOURCE_CONTAINER:-dssm-recall}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zcx/pairec-python-runtime}"
ARCHIVE="${ARCHIVE:-/tmp/pairec-pymilvus-runtime-$$.tar.gz}"
CONTAINER_ARCHIVE="/tmp/pairec-pymilvus-runtime-$$.tar.gz"

command -v docker >/dev/null || { echo "ERROR: docker is required" >&2; exit 1; }
docker inspect "$SOURCE_CONTAINER" >/dev/null 2>&1 || {
  echo "ERROR: source container does not exist: $SOURCE_CONTAINER" >&2
  exit 1
}

echo "== Export Python distributions from $SOURCE_CONTAINER =="
docker exec "$SOURCE_CONTAINER" env -u LD_PRELOAD python -c \
  'import grpc, pymilvus, ujson; print("SOURCE_PYMILVUS_RUNTIME_OK", pymilvus.__version__, grpc.__version__)'
docker exec -i "$SOURCE_CONTAINER" env -u LD_PRELOAD python - \
  "$CONTAINER_ARCHIVE" pymilvus grpcio ujson milvus-lite <<'PY'
import importlib.metadata
import pathlib
import sys
import tarfile

archive = sys.argv[1]
names = sys.argv[2:]
added = set()
with tarfile.open(archive, "w:gz") as output:
    for name in names:
        distribution = importlib.metadata.distribution(name)
        root = pathlib.Path(distribution.locate_file(""))
        print(f"distribution={name} version={distribution.version} root={root}")
        for entry in distribution.files or []:
            source = pathlib.Path(distribution.locate_file(entry))
            if not source.exists() or source.is_dir():
                continue
            relative = source.relative_to(root)
            key = str(relative)
            if key in added:
                continue
            output.add(source, arcname=key, recursive=False)
            added.add(key)
print(f"archive={archive} files={len(added)}")
PY

docker cp "$SOURCE_CONTAINER:$CONTAINER_ARCHIVE" "$ARCHIVE"
docker exec "$SOURCE_CONTAINER" rm -f "$CONTAINER_ARCHIVE"
mkdir -p "$OUTPUT_DIR"
tar -xzf "$ARCHIVE" -C "$OUTPUT_DIR"
rm -f "$ARCHIVE"

test -d "$OUTPUT_DIR/pymilvus" || {
  echo "ERROR: pymilvus package was not exported" >&2
  exit 1
}

echo "PAIREC_PYMILVUS_RUNTIME_EXPORT_OK output_dir=$OUTPUT_DIR"
