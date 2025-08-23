# app/benchmark_logger.py
# Orchestrates running batch + stream over a prompt set and logging metrics.

import app.runtime_setup  # keep first to quiet warnings/env

from datetime import datetime, timezone
from typing import Iterable
import random

from app.config_loader import load_config
from app.prompt_loader import load_prompts
from app.model_loader import load_model
from app.inference_batch import run_batch
from app.inference_stream import run_stream
from app.metrics_logger import log_row

from app.logger_config import get_logger   # add logging
log = get_logger(__name__)                 # module logger


def utc_now_iso() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    log.debug(f"utc_now_iso() -> {ts}")   # debug-level (useful for tracing, not noisy)
    return ts

def iter_prompts(df, sample_size: int | None) -> Iterable[dict]:
    """
    Returns an iterator of prompt rows as dicts.
    If sample_size is provided (>0), take the first N rows for determinism.
    """
    if sample_size and sample_size > 0:
        df = df.head(sample_size)
    for _, row in df.iterrows():
        yield {"prompt_id": str(row["prompt_id"]), "prompt_text": str(row["prompt_text"])}


def run_benchmark():
    """
    Load config/model/prompts and run both modes (batch + stream)
    for each prompt, repeating num_runs times, logging every run to CSV.
    """
    log.info("start run_benchmark function")
    cfg = load_config()
    model_id = cfg["model_id"]
    device_cfg = cfg.get("device", "cpu")
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    temperature = float(cfg.get("temperature", 0.7))
    num_runs = int(cfg.get("num_runs", 1))
    sample_size = int(cfg.get("prompt_sample_size", 0)) or None

    # Load model once
    tokenizer, model, device = load_model(model_id, device_cfg)

    # Load prompts
    df = load_prompts()
    prompts = list(iter_prompts(df, sample_size))

    total_runs = 0
    for p in prompts:
        prompt_id = p["prompt_id"]
        prompt_text = p["prompt_text"]

        for r in range(num_runs):
            # --- Batch ---
            batch_res = run_batch(
                tokenizer, model, prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            log_row({
                "timestamp": utc_now_iso(),
                "mode": "batch",
                "prompt_id": prompt_id,
                "model_id": model_id,
                "device": device,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "total_latency_ms": batch_res["total_latency_ms"],
                "tokens_per_sec": batch_res["tokens_per_sec"],
                "output_len": batch_res["output_len"],
            })

            # --- Stream ---
            stream_res = run_stream(
                tokenizer, model, prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            log_row({
                "timestamp": utc_now_iso(),
                "mode": "stream",
                "prompt_id": prompt_id,
                "model_id": model_id,
                "device": device,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "total_latency_ms": stream_res["total_latency_ms"],
                "tokens_per_sec": stream_res["tokens_per_sec"],
                "output_len": stream_res["output_len"],
            })

            total_runs += 2  # batch + stream

    log.info("End benchmark function and print out should be below")
    return {
        "prompts": len(prompts),
        "num_runs_per_prompt": num_runs,
        "total_logged_rows": total_runs,
        "model_id": model_id,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }


# ---- Built-in smoke test (consistent with other modules) ----
if __name__ == "__main__":
    summary = run_benchmark()
    print("\n" + "=" * 60 + "\n")
    print({
        "prompts": summary["prompts"],
        "num_runs_per_prompt": summary["num_runs_per_prompt"],
        "total_logged_rows": summary["total_logged_rows"],
        "model_id": summary["model_id"],
        "device": summary["device"],
    })
    print("\nWrote rows to data/inference_logs.csv\n")


# Run python3 -m app.benchmark_logger
