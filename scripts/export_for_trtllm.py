#!/usr/bin/env python
"""Export fine-tuned Qwen3 checkpoint to HuggingFace format for TRT-LLM."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.decoder.qwen3_generative_rec import Qwen3GenerativeRec

CKPT = "./checkpoints/decoder_qwen3/decoder_epoch_15.pt"   # or epoch_20
OUTDIR = "./exported/qwen3_rec"

print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu")
cfg = ckpt.get("config", {})

print("Creating model...")
model = Qwen3GenerativeRec(
    model_name_or_path="./models/Qwen3-0.6B",
    vocab_size=cfg.get("vocab_size", 256),
    num_quantizers=cfg.get("num_quantizers", 4),
    max_seq_len=cfg.get("max_seq_len", 2048),
    use_lora=False,
)
model.load_state_dict(ckpt["model_state_dict"], strict=False)

# Merge LoRA and export
model.merge_lora()
os.makedirs(OUTDIR, exist_ok=True)
model.base_model.save_pretrained(OUTDIR, safe_serialization=True)
model.tokenizer.save_pretrained(OUTDIR)

# Fix vocab_size in config (resize_token_embeddings might not update it)
import json
with open(f"{OUTDIR}/config.json", "r") as f:
    config = json.load(f)
config["vocab_size"] = len(model.tokenizer)
with open(f"{OUTDIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Exported to {OUTDIR}")
print(f"vocab_size={config['vocab_size']}  hidden_size={config.get('hidden_size')}")
print(f"Next: python TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py --model_dir {OUTDIR} --output_dir ./trt_ckpt --dtype bfloat16")
