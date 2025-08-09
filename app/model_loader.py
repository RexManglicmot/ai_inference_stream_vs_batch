# app/model_loader.py
import app.runtime_setup  # keep this first

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # safe on Mac

import torch
torch.set_num_threads(1)

from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config_loader import load_config

def pick_device(cfg_device: str) -> str:
    # If user asked for mps, use it if available
    if cfg_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    # If user asked for cpu but MPS exists, prefer mps to avoid CPU-native crash
    if cfg_device == "cpu" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_id: str, device: str = "cpu") -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    device = pick_device(device)
    # print(f"[device] using: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS is happier with float16; CPU stays float32
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    # sometimes helps on mac backends
    model.config.use_cache = False
    return tokenizer, model, device

if __name__ == "__main__":
    cfg = load_config()
    tok, mdl, device = load_model(cfg["model_id"], cfg.get("device", "cpu"))

    prompt = "Transformer caching lets models reuse key/value states, which"
    inputs = tok(prompt, return_tensors="pt").to(device)

    max_tokens = min(int(cfg.get("max_new_tokens", 64)), 12)  # very small for smoke test
    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,            # ← greedy
            # temperature is ignored when do_sample=False
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n[MODEL OUTPUT]\n", text[:300])
