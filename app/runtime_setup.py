# app/runtime_setup.py
# Central place to silence noisy warnings & progress bars.

import os, warnings

# Quiet HF downloads/progress + tokenizer threads
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Mac (MPS) fallback + smaller thread counts
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Hide specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MPS backend.*fall back to run on the CPU.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# Silence Transformers’ internal logger
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
