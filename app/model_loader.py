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
    
    Args:
        model_id: Name or path of the HF model ("gpt2")
        device: Device to rund the mon on ("cpu")
        
    Returns:
        tokenizer: Tokenizer object that converts text into tokens
        model: Pretrained model ready for inference
        device: Device stirng used for model placement
    """

    # Load tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    # pad token=?

    # Ensure tokenizer has a pad token to avoid warnings during batching
    if tokenizer.pad_token is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model weights into memory
    # Need to review args here
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Move model to CPU
    model.to(device)

    # Set to evaluation mode (no training gradients)
    model.eval()

    return tokenizer, model, device

if __name__ == "__main__":
    # Smoke test to confirm that the model loads and can generate text

    # Load the configuration from YAML
    cfg = load_config()

    # Load tokenizer and model based on settings
    tok, mdl, device = load_model(cfg['model_id'], cfg.get("device", "cpu"))

    # Define a simple prompt
    prompt = "Say one short sentence about transformer caching."

    # Tokenize prompt and move tensors to correct device
    inputs = tok(prompt, return_tensors="pt").to(device)

    # Run the model in inference mode (no gradient calc)
    with torch.no_grad():
        out = m


