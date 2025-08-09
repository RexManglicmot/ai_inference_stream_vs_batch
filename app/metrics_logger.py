# app/metrics_logger.py
# Append benchmark metrics to data/inference_logs.csv (creates file + header if missing)

from pathlib import Path
import csv
from datetime import datetime

DEFAULT_PATH = Path("data/inference_logs.csv")

FIELDS = [
    "timestamp",
    "mode",                 # "batch" or "stream"
    "prompt_id",
    "model_id",
    "device",
    "max_new_tokens",
    "temperature",
    "total_latency_ms",
    "tokens_per_sec",
    "output_len",
]

def ensure_header(path: Path = DEFAULT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def log_row(row: dict, path: Path = DEFAULT_PATH):
    """row must include keys in FIELDS (extras are ignored)."""
    ensure_header(path)
    filtered = {k: row.get(k, "") for k in FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(filtered)

if __name__ == "__main__":
    # minimal fake row just to prove it writes correctly
    log_row({
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mode": "batch",
        "prompt_id": "TEST-001",
        "model_id": "distilgpt2",
        "device": "mps",
        "max_new_tokens": 32,
        "temperature": 0.7,
        "total_latency_ms": 1234,
        "tokens_per_sec": 20.5,
        "output_len": 32,
    })
    print("Wrote one row to data/inference_logs.csv")
