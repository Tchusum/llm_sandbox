"""Module for tokenization and detokenization of text using a tokenizer."""
import tiktoken
import torch


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Convert raw text to a tensor of token IDs using the provided tokenizer.

    :param text: The input text string to be tokenized.
    :param tokenizer: An instance of tiktoken.Encoding used for tokenization.
    :return: A torch.Tensor containing the token IDs, with a batch dimension added.
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0) # add batch dimension


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Convert a tensor of token IDs back to text using the provided tokenizer.

    :param token_ids: A torch.Tensor containing the token IDs, with a batch dimension.
    :param tokenizer: An instance of tiktoken.Encoding used for detokenization.
    :return: The decoded text string.
    """
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())
