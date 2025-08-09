# app/config_loader.py

import yaml
from pathlib import Path

def load_config(path: str = "config/settings.yaml") -> dict:
    """
    Load YAML config and return a dict
    Keeps parameters (model_id, device, num_rums, etc.) outside the code
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

if __name__ == "__main__":
    # Smoke test: prints the config"
    cfg = load_config()
    print(cfg)

    # It works and prints in the terminal
    