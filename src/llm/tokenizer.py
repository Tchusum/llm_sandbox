import tiktoken
import torch
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding | Qwen3Tokenizer) -> torch.Tensor:

    if isinstance(tokenizer, Qwen3Tokenizer):
        encoded = tokenizer.encode(text)
    elif isinstance(tokenizer, tiktoken.Encoding):
        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    else:
        msg = "Unsupported tokenizer type. Must be Qwen3Tokenizer or tiktoken.Encoding."
        raise TypeError(msg)

    return torch.tensor(encoded).unsqueeze(0) # add batch dimension


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Convert a tensor of token IDs back to text using the provided tokenizer.

    :param token_ids: A torch.Tensor containing the token IDs, with a batch dimension.
    :param tokenizer: An instance of tiktoken.Encoding used for detokenization.
    :return: The decoded text string.
    """
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())
