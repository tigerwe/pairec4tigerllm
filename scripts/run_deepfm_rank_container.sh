#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-deepfm-rank}"
IMAGE="${IMAGE:-docker.io/library/zcx-pairec-image:v1.1}"
REPO_DIR="${REPO_DIR:-/home/zcx/workspace/pairec4tigerllm}"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/deepfm_out}"
PORT="${PORT:-18210}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-8g}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-60}"
MODEL_ROLE="${MODEL_ROLE:-engineering}"

case "$MODEL_ROLE" in
  engineering|production_candidate) ;;
  *) echo "ERROR: MODEL_ROLE must be engineering or production_candidate" >&2; exit 1 ;;
esac

for path in \
  "$REPO_DIR/inference/deepfm_rank_server.py" \
  "$MODEL_DIR/deepfm_best.pt" \
  "$MODEL_DIR/feature_vocab.json" \
  "$MODEL_DIR/model_config.json" \
  "$MODEL_DIR/training_summary.json" \
  "$MODEL_DIR/user_profiles.json" \
  "$MODEL_DIR/item_categories.json"; do
  test -f "$path" || { echo "ERROR: missing runtime file: $path" >&2; exit 1; }
done

docker stop --time 10 "$CONTAINER" >/dev/null 2>&1 || true
docker rm "$CONTAINER" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER" \
  --network host \
  --restart unless-stopped \
  --cpus "$CPUS" \
  --memory "$MEMORY" \
  -e DEEPFM_MODEL_PATH=/models/deepfm/deepfm_best.pt \
  -e DEEPFM_VOCAB_PATH=/models/deepfm/feature_vocab.json \
  -e DEEPFM_PROFILES_PATH=/models/deepfm/user_profiles.json \
  -e DEEPFM_CATEGORIES_PATH=/models/deepfm/item_categories.json \
  -e DEEPFM_EXPECTED_CANDIDATES=50 \
  -e DEEPFM_MODEL_ROLE="$MODEL_ROLE" \
  -e DEEPFM_RANK_PORT="$PORT" \
  -e DEEPFM_DEVICE=cpu \
  -v "$REPO_DIR:/workspace:ro" \
  -v "$MODEL_DIR:/models/deepfm:ro" \
  --workdir /workspace \
  --entrypoint /bin/bash \
  "$IMAGE" \
  --noprofile --norc -lc \
  'unset LD_PRELOAD HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; exec python /workspace/inference/deepfm_rank_server.py'

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 \
      "http://127.0.0.1:${PORT}/health" >/tmp/deepfm-rank-health.json; then
    python - /tmp/deepfm-rank-health.json "$MODEL_ROLE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("code") == 200 and data.get("status") == "healthy", data
assert data.get("expected_candidates") == 50, data
assert data.get("model_role") == sys.argv[2], data
assert data.get("checkpoint_epoch", 0) > 0, data
assert all(data.get("vocab_sizes", {}).get(name, 0) > 0
           for name in ("user", "item", "cat", "gender", "age")), data
print("DEEPFM_RANK_HEALTH_OK model_version={} model_role={} checkpoint_epoch={}".format(
    data["model_version"], data["model_role"], data["checkpoint_epoch"]))
PY
    echo "DEEPFM_RANK_CONTAINER_OK container=${CONTAINER} endpoint=127.0.0.1:${PORT} model_role=${MODEL_ROLE}"
    exit 0
  fi
  if ! docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
    docker logs "$CONTAINER" >&2 || true
    echo "ERROR: DeepFM rank container exited" >&2
    exit 1
  fi
  sleep 1
done

docker logs "$CONTAINER" >&2 || true
echo "ERROR: DeepFM rank service did not become healthy" >&2
exit 1
