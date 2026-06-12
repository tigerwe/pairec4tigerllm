#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-deploy/inference}"

kubectl -n pairec exec "$TARGET" -- bash -lc '
set -euo pipefail

echo "== Engine config =="
python - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

engine_dir = Path(os.environ.get("TRT_ENGINE_DIR", "/app/trt_engines/qwen3_rec_v4"))
config_path = engine_dir / "config.json"
print("engine_dir:", engine_dir)
print("config:", config_path)
data = json.load(open(config_path))
keys = {
    "context_fmha",
    "use_paged_context_fmha",
    "paged_kv_cache",
    "kv_cache_type",
    "compute_context_logits",
    "max_input_len",
    "max_seq_len",
    "max_num_tokens",
}

def walk(obj, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                print(f"{path}/{key} = {value}")
            walk(value, f"{path}/{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            walk(value, f"{path}[{idx}]")

walk(data)
PY

echo
echo "== TensorRT-LLM Python path =="
python - <<'"'"'PY'"'"'
import inspect
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunnerCpp

print("tensorrt_llm:", tensorrt_llm.__file__)
print("version:", getattr(tensorrt_llm, "__version__", "unknown"))
print("ModelRunnerCpp.from_dir:", inspect.signature(ModelRunnerCpp.from_dir))
PY

echo
echo "== DataSystem C++ symbols/strings =="
found=0
while IFS= read -r so; do
  if strings "$so" | grep -Eq "TensorRT-LLM.*Datasystem|op=offload|op=onboard|Create Datasystem class|Init KvCache Manager DataSystem"; then
    found=1
    echo "---- $so"
    strings "$so" \
      | grep -E "TensorRT-LLM.*Datasystem|op=offload|op=onboard|Create Datasystem class|Init KvCache Manager DataSystem|KV cache reuse disabled" \
      | head -120
  fi
done < <(find /home/TensorRT-LLM /usr/local/lib64/python3.11/site-packages /usr/local/lib/python3.11/site-packages -name "*.so" 2>/dev/null)

if [ "$found" -eq 0 ]; then
  echo "NO_DATASYSTEM_CPP_STRINGS_FOUND"
fi
'
