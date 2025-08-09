# app/inference_stream.py
# Stream-like (token-by-token) generation with basic metrics.
# Matches the style of inference_batch.py
import app.runtime_setup  # keep this first

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # allow CPU fallback for missing MPS ops

import time
import torch


def run_stream(tokenizer, model, prompt: str, *, max_new_tokens: int, temperature: float, device: str):
    """
    Generate token-by-token (greedy) and measure:
      - total_latency_ms
      - tokens_per_sec
      - output_len
    Notes:
      - Per-step overhead makes stream slower than batch; that’s expected.
      - Greedy decoding keeps results stable for benchmarking.
    """
    # Encode prompt once
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # Prime KV cache on the prompt
    with torch.no_grad():
        prime = model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=True, return_dict=True)
    past_kv = prime.past_key_values

    generated_chunks = []
    next_input = input_ids[:, -1:]  # start from last token of the prompt

    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            step = model(input_ids=next_input, use_cache=True,
                         past_key_values=past_kv, return_dict=True)
        logits = step.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
        generated_chunks.append(next_token)
        past_kv = step.past_key_values
        next_input = next_token
    total_s = time.perf_counter() - t0

    # Stitch tokens back together
    if generated_chunks:
        gen_ids = torch.cat(generated_chunks, dim=1)
        full_ids = torch.cat([input_ids, gen_ids.to(input_ids.device)], dim=1)
    else:
        full_ids = input_ids

    text = tokenizer.decode(full_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    gen_tokens = full_ids.shape[1] - input_ids.shape[1]
    tps = gen_tokens / total_s if total_s > 0 else 0.0

    return {
        "text": text,
        "total_latency_ms": int(total_s * 1000),
        "tokens_per_sec": tps,
        "output_len": gen_tokens,
    }


# ---- Built-in smoke test (consistent with other modules) ----
if __name__ == "__main__":
    from app.config_loader import load_config
    from app.model_loader import load_model

    cfg = load_config()
    tok, mdl, device = load_model(cfg["model_id"], cfg.get("device", "cpu"))

    prompt = "Transformer caching lets models reuse key/value states, which"
    res = run_stream(
        tok, mdl, prompt,
        max_new_tokens=min(32, cfg.get("max_new_tokens", 32)),
        temperature=cfg.get("temperature", 0.7),
        device=device,
    )
    print("\n")
    print({
        "total_latency_ms": res["total_latency_ms"],
        "tokens_per_sec": round(res["tokens_per_sec"], 3),
        "output_len": res["output_len"],
    })
    print("\n")
    print(res["text"][:200])
    print("\n") 
