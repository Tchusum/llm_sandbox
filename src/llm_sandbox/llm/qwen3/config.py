from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer

tokenizer_path = Path("data/qwen3/tokenizer-base.json")
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)


def get_model_schema_qwen3() -> dict:
    """Used to avoid circular imports when importing GPTModel in llm_call.py."""
    from llm_sandbox.llm.models import LLMQwen3Model
    return {
        "tokenizer": Qwen3Tokenizer(tokenizer_file_path=tokenizer_path),
        "model": LLMQwen3Model(),
        "name": "qwen3-0.6B-base.pth",
        "eos_id": tokenizer.eos_token_id,
    }


class QWENConfig06B(BaseModel):
    vocab_size: int = 151_936
    context_length: int = 40_960
    emb_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 28
    hidden_dim: int = 3072
    head_dim: int = 128
    qk_norm: bool = True
    n_kv_groups: int = 8
    rope_base: float = 1_000_000.0
    dtype: Any = torch.bfloat16
