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
from app.logger_config import get_logger   

# module-level logger
log = get_logger(__name__)            

def pick_device(cfg_device: str) -> str:
    """
    Function to pick device based on order
    """
    if cfg_device == "mps" and torch.backends.mps.is_available():
        log.info("Requested mps --> using MPS backend")
        return "mps"
    if cfg_device == "cpu" and torch.backends.mps.is_available():
        log.info("Requested cpu but MPS is still available bro --> using MPS for stability")
        return "mps"
    
    log.info("Defaulting to CPU")
    return "cpu"


def load_model(model_id: str, device: str = "cpu") -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    device = pick_device(device)

    # Log device
    log.info(f"Using device: {device}")                       

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        # Debug-level detail
        log.debug("Set tokenizer.pad_token to eos_token")     

    dtype = torch.float16 if device == "mps" else torch.float32
    log.info(f"Loading model '{model_id}' with dtype={dtype}")  # log dtype/model

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.config.use_cache = False  # helps some mac backends
    log.info("Model loaded and ready (eval mode, use_cache=False)")  # success
    return tokenizer, model, device


if __name__ == "__main__":
    cfg = load_config()
    tok, mdl, device = load_model(cfg["model_id"], cfg.get("device", "cpu"))

    prompt = "Transformer caching lets models reuse key/value states, which"
    inputs = tok(prompt, return_tensors="pt").to(device)

    max_tokens = min(int(cfg.get("max_new_tokens", 64)), 12)
    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n[MODEL OUTPUT]\n", text[:300])

    # Run python3 -m app.model_loader

    # MPS is used
    # Model is distilgpt2 with float16
    # Model is loaded
    # It worked
