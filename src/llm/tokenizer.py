"""Tokenizer module for GPT-style language model training."""
import logging
from pathlib import Path

import requests
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

TOKENIZER_VOCAB_SIZE = 50257

def extract_text_data() -> str:
    """Download and return the text content of 'the-verdict.txt'.

    If the file does not exist locally, fetches it from the rasbt
    LLMs-from-scratch GitHub repository and saves it to disk.

    Returns:
        str: The full text content of the file.

    Raises:
        requests.HTTPError: If the HTTP request to download the file fails.

    """
    if not Path("./data/the-verdict.txt").exists():
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        file_path = "./data/the-verdict.txt"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        Path(file_path).write_bytes(response.content)

    return Path("./data/the-verdict.txt").read_text(encoding="utf-8")


class GPTDataset(Dataset):
    """A PyTorch Dataset for GPT-style language model training.

    Tokenizes the input text and uses a sliding window to produce overlapping
    input/target sequence pairs suitable for next-token prediction training.

    Attributes:
        input_ids (list[torch.Tensor]): List of input token ID tensors, each of
            shape ``(max_length,)``.
        target_ids (list[torch.Tensor]): List of target token ID tensors, each of
            shape ``(max_length,)``, offset by one position from the corresponding
            input.

    """

    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int) -> None:
        """Initialize GPTDataset by tokenizing text and building sliding windows.

        Args:
            txt (str): Raw input text to tokenize.
            tokenizer (tiktoken.Encoding): Tokenizer used to encode the text.
            max_length (int): Length of each input/target token sequence.
            stride (int): Step size for the sliding window over the token sequence.

        Raises:
            AssertionError: If the number of tokens is not greater than ``max_length``.

        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        if len(token_ids) <= max_length:
            error_msg = (f"""
                Number of tokenized inputs ({len(token_ids)}) must be greater than max_length ({max_length}).
            """)
            raise ValueError(error_msg)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Total number of input/target sequence pairs.

        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input/target pair at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair of tensors
                ``(input_ids, target_ids)``, each of shape ``(max_length,)``.

        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt: str,
    batch_size: int,
    max_length: int,
    stride: int,
) -> DataLoader:
    """Create a DataLoader for GPT-style language model training.

    Tokenizes the input text using the GPT-2 tokenizer and wraps it in a
    sliding-window dataset, then returns a PyTorch DataLoader over that dataset.

    Args:
        txt (str): Raw text to tokenize and load.
        batch_size (int): Number of samples per batch.
        max_length (int): Length of each input/target token sequence.
        stride (int): Step size for the sliding window over the token sequence.

    Returns:
        torch.utils.data.DataLoader: DataLoader yielding ``(input_ids, target_ids)``
            pairs, each of shape ``(batch_size, max_length)``.

    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
    )


def build_embeddings(
    output_path: str,
    max_length: int = 256,
    stride: int = 128,
    batch_size: int = 8,
    output_dim: int = 256,
) -> torch.Tensor:
    """Build and saves token + positional embeddings from raw text data.

    Extracts text, tokenizes it into sliding-window sequences, and passes each
    batch through freshly initialized token and positional embedding layers.
    The resulting embeddings are concatenated and persisted to disk.

    Args:
        output_path (str): File path where the embedding tensor will be saved
            (via ``torch.save``).
        max_length (int, optional): Sequence length for each token window.
            Defaults to 256.
        stride (int, optional): Step size between consecutive windows.
            Defaults to 128.
        batch_size (int, optional): Number of sequences per batch.
            Defaults to 8.
        output_dim (int, optional): Embedding dimensionality for both token
            and positional embeddings. Defaults to 256.

    Returns:
        torch.Tensor: Concatenated input embeddings of shape
            ``(num_sequences, max_length, output_dim)``.

    """
    text_data = extract_text_data()

    dataloader = create_dataloader(
        text_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
    )

    token_embedding_layer = torch.nn.Embedding(TOKENIZER_VOCAB_SIZE, output_dim)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    all_embeddings = []
    for inputs, _ in dataloader:
        token_embeddings = token_embedding_layer(inputs)
        input_embeddings: torch.Tensor = token_embeddings + pos_embeddings
        # Detach to avoid retaining the computation graph and save memory
        all_embeddings.append(input_embeddings.detach())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, output_path)
    logger.info("Saved embeddings of shape %s to %s", all_embeddings.shape, output_path)
    return all_embeddings


def text_to_token_ids(text: str, tokenizer: object) -> torch.Tensor:
    """Convert raw text to a tensor of token IDs using the provided tokenizer."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0) # add batch dimension


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: object) -> str:
    """Convert a tensor of token IDs back to text using the provided tokenizer."""
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

if __name__ == "__main__":
    path = "/workspaces/llm_sandbox/data/embeddings.pt"
    embeddings = build_embeddings(path)
