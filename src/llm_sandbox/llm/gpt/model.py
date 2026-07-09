import torch
from torch import nn

from llm_sandbox.llm.gpt.attention import MultiHeadAttention, MultiHeadAttentionConfig
from llm_sandbox.llm.gpt.config import GPTConfig


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
