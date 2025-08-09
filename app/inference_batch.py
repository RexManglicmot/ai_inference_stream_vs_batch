# app/inference_batch.py
# Full (non-streaming) generation with basic metrics.
import app.runtime_setup  # keep this first

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # allow CPU fallback for missing MPS ops


import time
import torch

def run_batch(tokenizer, model, prompt: str, *, max_new_tokens: int, temperature: float, device: str):
    """
    Execute full generation and measure:
      - total_latency_ms
      - tokens_per_sec
      - output_len
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic & stable for benchmarking
        )
    total_s = time.perf_counter() - t0

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    tps = gen_tokens / total_s if total_s > 0 else 0.0

    return {
        "text": text,
        "total_latency_ms": int(total_s * 1000),
        "tokens_per_sec": tps,
        "output_len": gen_tokens,
    }

# ---- Built-in smoke test (consistent with model_loader.py) ----
if __name__ == "__main__":
    from app.config_loader import load_config
    from app.model_loader import load_model

    cfg = load_config()
    tok, mdl, device = load_model(cfg["model_id"], cfg.get("device", "cpu"))

    prompt = "Transformer caching lets models reuse key/value states, which"
    res = run_batch(
        tok, mdl, prompt,
        max_new_tokens=min(32, cfg.get("max_new_tokens", 32)),
        temperature=cfg.get("temperature", 0.7),
        device=device,
    )

    print({"total_latency_ms": res["total_latency_ms"],
           "tokens_per_sec": round(res["tokens_per_sec"], 3),
           "output_len": res["output_len"]})
    print("\n") 
    print(res["text"][:200].strip())        # removes extra spaces/newlines
    print("\n") 
