from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from reasoning_from_scratch.qwen3 import QWEN_CONFIG_06_B, Qwen3Model, download_qwen3_small

from llm_sandbox.llm.gpt.config import MODEL_CONFIG
from llm_sandbox.llm.gpt.model import GPTConfig, GPTModel
from llm_sandbox.llm.qwen3.config import QWENConfig06B
from llm_sandbox.llm.tokenizer import text_to_token_ids, token_ids_to_text

if TYPE_CHECKING:
    import tiktoken


class LLMModel(ABC):

    @abstractmethod
    def load(self, model_name: str, device: torch.device) -> None:
        pass

class LLMGPTModel(LLMModel):

    def load(self, model_name: str, device: torch.device, base_key: str | None = None) -> None:

        model_path_input = Path("data/gpt2")

        # Resolve checkpoint filename
        model_path = Path(__file__).parent.parent.parent.parent / model_path_input / model_name
        if not model_path.exists():
            msg = "Model file not found."
            raise FileNotFoundError(msg)

        # Resolve base model key from MODEL_CONFIG
        for key in MODEL_CONFIG:
            if model_name.startswith(key):
                base_key = key
                break
        if base_key is None:
            msg = f"Base model key not found for {model_name}."
            raise ValueError(msg)

        # Instantiate model using the resolved base model configuration
        config = GPTConfig(
            emb_dim=MODEL_CONFIG[base_key]["emb_dim"],
            n_layers=MODEL_CONFIG[base_key]["n_layers"],
            n_heads=MODEL_CONFIG[base_key]["n_heads"],
        )
        model = GPTModel(config)

        # Load weights and prepare model
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            ),
        )
        model.to(device)
        model.eval()

        self.model = model
        self.config = config


class LLMQwen3Model(LLMModel):

    def __init__(self) -> None:
        self.config = QWENConfig06B(**QWEN_CONFIG_06_B)

    def load(self, model_name: str, device: torch.device) -> None:

        model_path_input = Path("data/qwen3")

        download_qwen3_small(kind="base", tokenizer_only=False, out_dir=model_path_input)

        model_path = model_path_input / model_name
        model = Qwen3Model(self.config.model_dump())

        # Load weights and prepare model
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            ),
        )
        model.to(device)
        model.eval()

        self.model = model


@torch.inference_mode()
def generate(
    model_instance: LLMModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    *,
    exclude_input: bool = False,
) -> torch.Tensor:
    """Generate text from the model given an initial context.

    :param model_instance: The LLM model instance
    :param idx: Input tensor containing token indices
    :param max_new_tokens: Maximum number of new tokens to generate
    :param context_size: Size of the context window to consider for generation
    :param temperature: Temperature for sampling (default: 0.0)
    :param top_k: If specified, only consider the top_k logits for sampling (default: None)
    :param eos_id: If specified, stop generation when this token ID is generated
    :return: Tensor containing the generated token indices
    """
    # remember original input length (to optionally exclude it from output)
    orig_len = idx.shape[1]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model_instance.model(idx_cond)[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)  # shape (batch, 1)
            mask = logits < min_val
            logits = logits.masked_fill(mask, float("-inf"))

        if temperature > 0.0:
            logits = logits / temperature

            # Numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # check if the generated token is the end-of-sequence token
        if eos_id is not None and (idx_next == eos_id).all():
            break

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    if exclude_input:
        return idx[:, orig_len:]
    return idx


def generate_and_print(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
    eos_id: int | None = None,
) -> None:
    """Generate a sample text from the model and print it.

    :param model: The GPT model used for generating text.
    :param tokenizer: The tokenizer used for encoding and decoding text.
    :param device: The device (CPU or GPU) to run the generation on.
    :param start_context: The initial context string for generating sample text.
    :param eos_id: Optional end-of-sequence token ID to stop generation early.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            eos_id=eos_id,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
