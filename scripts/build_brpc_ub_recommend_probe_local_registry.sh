#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc-827}
LOCAL_BCR_REGISTRY=${LOCAL_BCR_REGISTRY:-/home/zcx/bazel-local-registry/bcr}
LOCAL_SECRET_REGISTRY=${LOCAL_SECRET_REGISTRY:-/home/zcx/bazel-local-registry/secretflow}
BAZEL_OUTPUT_BASE=${BAZEL_OUTPUT_BASE:-/root/.cache/bazel/_bazel_root/0947eeff3cdbdab635f34a3b3ff5f6d1}
MODULE_GRAPH_OUT=${MODULE_GRAPH_OUT:-/tmp/brpc-module-graph.txt}
RUN_BUILD=${RUN_BUILD:-1}
OPENSSL_VERSION=${OPENSSL_VERSION:-3.3.2.bcr.1}

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

registry_uri()
{
    local path=$1
    [[ "$path" == /* ]] || fail "registry path must be absolute: $path"
    printf 'file://%s' "${path%/}"
}

[[ -d "$BRPC_ROOT" ]] || fail "bRPC root does not exist: $BRPC_ROOT"
[[ -f "$BRPC_ROOT/MODULE.bazel" ]] || fail "missing $BRPC_ROOT/MODULE.bazel"

required_registry_files=(
    "$LOCAL_BCR_REGISTRY/bazel_registry.json"
    "$LOCAL_SECRET_REGISTRY/bazel_registry.json"
    "$LOCAL_SECRET_REGISTRY/modules/leveldb/1.23/MODULE.bazel"
    "$LOCAL_SECRET_REGISTRY/modules/leveldb/1.23/source.json"
    "$LOCAL_SECRET_REGISTRY/modules/openssl/$OPENSSL_VERSION/MODULE.bazel"
    "$LOCAL_SECRET_REGISTRY/modules/openssl/$OPENSSL_VERSION/source.json"
)
for file in "${required_registry_files[@]}"; do
    [[ -f "$file" ]] || fail "missing local registry file: $file"
done

secret_registry_uri=$(registry_uri "$LOCAL_SECRET_REGISTRY")
bcr_registry_uri=$(registry_uri "$LOCAL_BCR_REGISTRY")
module_file="$BRPC_ROOT/MODULE.bazel"
backup_file="$BRPC_ROOT/MODULE.bazel.before-local-registry"
if [[ ! -e "$backup_file" ]]; then
    cp -p "$module_file" "$backup_file"
fi

python3 - "$module_file" "$secret_registry_uri" "$OPENSSL_VERSION" <<'PY'
import pathlib
import re
import sys

module_path = pathlib.Path(sys.argv[1])
registry_uri = sys.argv[2]
openssl_version = sys.argv[3]
lines = module_path.read_text().splitlines(keepends=True)
targets = {"leveldb", "openssl"}
updated = set()
openssl_dependency_updated = False
in_override = False
current_module = None

for index, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("bazel_dep(") and re.search(r'name\s*=\s*["\']openssl["\']', stripped):
        replacement, count = re.subn(
            r'(version\s*=\s*)["\'][^"\']+["\']',
            rf'\1"{openssl_version}"',
            line,
            count=1,
        )
        if count != 1:
            raise SystemExit("unable to update openssl bazel_dep version")
        lines[index] = replacement
        openssl_dependency_updated = True
        continue
    if stripped == "single_version_override(":
        in_override = True
        current_module = None
        continue
    if not in_override:
        continue

    module_match = re.match(r'module_name\s*=\s*["\']([^"\']+)["\']\s*,', stripped)
    if module_match:
        current_module = module_match.group(1)
        continue

    if current_module in targets and stripped.startswith("registry ="):
        indent = line[: len(line) - len(line.lstrip())]
        lines[index] = f'{indent}registry = "{registry_uri}",\n'
        updated.add(current_module)

    if stripped == ")":
        in_override = False
        current_module = None

missing = targets - updated
if missing:
    raise SystemExit("missing registry override(s): " + ", ".join(sorted(missing)))
if not openssl_dependency_updated:
    raise SystemExit("missing openssl bazel_dep")

module_path.write_text("".join(lines))
PY

if grep -Eq '\[file:|\]\(file:' "$module_file"; then
    fail "MODULE.bazel still contains Markdown-formatted registry values"
fi
override_count=$(grep -Fc "registry = \"$secret_registry_uri\"," "$module_file")
[[ "$override_count" == 2 ]] || fail "expected two local SecretFlow overrides, got $override_count"
grep -Eq "bazel_dep\(name = ['\"]openssl['\"], version = ['\"]$OPENSSL_VERSION['\"]" "$module_file" ||
    fail "openssl bazel_dep was not aligned to $OPENSSL_VERSION"

echo "BRPC_LOCAL_REGISTRY_CONFIG_OK module=$module_file backup=$backup_file openssl=$OPENSSL_VERSION"
echo "BRPC_LOCAL_REGISTRY_RC_ISOLATION_OK"

bazel_startup_args=()
bazel_startup_args+=("--ignore_all_rc_files")
if [[ -n "$BAZEL_OUTPUT_BASE" ]]; then
    bazel_startup_args+=("--output_base=$BAZEL_OUTPUT_BASE")
fi

(
    cd "$BRPC_ROOT"
    bazel "${bazel_startup_args[@]}" shutdown >/dev/null 2>&1 || true
    bazel "${bazel_startup_args[@]}" mod graph \
        --registry="$secret_registry_uri" \
        --registry="$bcr_registry_uri" \
        --ignore_dev_dependency \
        --lockfile_mode=off \
        >"$MODULE_GRAPH_OUT"
)

[[ -s "$MODULE_GRAPH_OUT" ]] || fail "module graph is empty: $MODULE_GRAPH_OUT"
echo "BRPC_LOCAL_REGISTRY_GRAPH_OK graph=$MODULE_GRAPH_OUT"

if [[ "$RUN_BUILD" == 0 ]]; then
    exit 0
fi

BRPC_ROOT="$BRPC_ROOT" \
BAZEL_OUTPUT_BASE="$BAZEL_OUTPUT_BASE" \
BAZEL_LOCKFILE_MODE=off \
BAZEL_IGNORE_ALL_RC_FILES=1 \
LOCAL_BCR_REGISTRY="$LOCAL_BCR_REGISTRY" \
LOCAL_SECRET_REGISTRY="$LOCAL_SECRET_REGISTRY" \
bash "$REPO_ROOT/scripts/build_brpc_ub_recommend_probe.sh"
