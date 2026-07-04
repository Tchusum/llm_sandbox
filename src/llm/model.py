"""A PyTorch implementation of a GPT-like language model.

Including the main components such as multi-head attention, feed-forward networks, and layer normalization.
"""
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn

from llm.attention import MultiHeadAttention, MultiHeadAttentionConfig
from llm.config import MODEL_CONFIG


class GPTConfig(BaseModel):
    """Configuration for the GPT-like language model."""

    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = True


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function."""

    def __init__(self) -> None:
        """Initialize the GELU activation function."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GELU activation function to the input tensor.

        :param x: Input tensor
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3)),
        ))


class FeedForward(nn.Module):
    """Feed-forward network consisting of two linear layers with GELU activation."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the feed-forward network.

        :param cfg: Configuration object containing model parameters
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        :param x: Input tensor
        """
        return self.layers(x)


class LayerNorm(nn.Module):
    """Layer normalization module."""

    def __init__(self, emb_dim: int) -> None:
        """Initialize the layer normalization module."""
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input tensor.

        :param x: Input tensor
        :return: Layer-normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class TransformerBlock(nn.Module):
    """A single transformer block consisting of multi-head attention and feed-forward network."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the transformer block.

        :param cfg: Configuration object containing model parameters
        """
        super().__init__()
        mha_cfg = MultiHeadAttentionConfig(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.att = MultiHeadAttention(mha_cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        :param x: Input tensor
        :return: Output tensor
        """
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut  # Add the original input back


class GPTModel(nn.Module):
    """GPT-like language model.

    The model consists of token and positional embeddings, multiple transformer blocks,
    and a final linear layer to produce logits for the vocabulary.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the GPT-like language model.

        :param cfg: Configuration object containing model parameters
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(
            cfg.emb_dim,
            cfg.vocab_size,
            bias=False,
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT-like language model."""
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def load_model(model_name: str, device: torch.device) -> tuple[GPTModel, GPTConfig]:
    """Load a GPT model and its weights using only `model_name`.

    `model_name` may be a base model (e.g. "gpt2-xl") or a checkpoint-like
    identifier (e.g. "gpt2-xl-alpaca-sft" or "my_checkpoint.pth"). The function
    resolves the checkpoint filename and the base model key from `MODEL_CONFIG`.

    :param model_name: Base model name or checkpoint-like name.
    :param device: The device to load the model onto.
    :return: The loaded GPTModel instance and its configuration.
    """
    # Resolve checkpoint filename
    name_file = f"{model_name}.pth"
    model_path = Path(__file__).parent.parent.parent / "data" / name_file
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
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def generate(  # noqa: PLR0913
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generate text from the model given an initial context.

    :param model: The GPT-like language model
    :param idx: Input tensor containing token indices
    :param max_new_tokens: Maximum number of new tokens to generate
    :param context_size: Size of the context window to consider for generation
    :param temperature: Temperature for sampling (default: 0.0)
    :param top_k: If specified, only consider the top_k logits for sampling (default: None)
    :param eos_id: If specified, stop generation when this token ID is generated
    :return: Tensor containing the generated token indices
    """
    # Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Apply temperature scaling
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

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


if __name__ == "__main__":

    # Create a sample batch of tokenized input data
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    # Implement the model
    torch.manual_seed(123)
    model = GPTModel(GPTConfig())

    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    # Test the text generation function
    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() # disable dropout

    out = generate(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPTConfig().context_length,
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
