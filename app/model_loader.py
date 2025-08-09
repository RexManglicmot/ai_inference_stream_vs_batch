# app/model_loader.py
# Loads a local Hugging Face model + tokenizer (No API  and keys)

from pathlib import Path
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# This one we created
# Loads project settings form settings.yaml
from app.config_loader import load_config

def load_model(model_id: str, device: str = "cpu") -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """
    Download/load a model + tokenizer locally via transformers
    Returns (tokenizer, model, device)
    """

