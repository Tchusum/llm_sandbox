
from pydantic import BaseModel

MODEL_CONFIG = {
    "gpt2-small": {"size": "124M", "emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"size": "355M", "emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {"size": "774M", "emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {"size": "1558M", "emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

 # End-of-sequence token ID for GPT-2 models
EOS_ID = 50256

class GPTConfig(BaseModel):
    """Configuration for the GPT-like language model."""

    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = True
