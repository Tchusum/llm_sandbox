"""Implement the `MultiHeadAttention` class, which is a key component of transformer-based models."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MultiHeadAttentionConfig:
    """Configuration for the MultiHeadAttention module."""

    d_in: int  # Input dimension (size of each input token embedding)
    d_out: int  # Output dimension (size of each output token embedding)
    context_length: int  # Maximum sequence length (context length)
    dropout: float  # Dropout probability for attention weights
    num_heads: int  # Number of attention heads
    qkv_bias: bool = False  # Whether to include bias terms in the linear layers for queries, keys, and values


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(
            self,
            cfg: MultiHeadAttentionConfig,
        ) -> None:
        """Initialize the multi-head attention module.

        :param cfg: Configuration object containing model parameters
        """
        super().__init__()

        self.d_out = cfg.d_out
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_out // cfg.num_heads

        self.W_query = nn.Linear(cfg.d_in, cfg.d_out, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(cfg.d_in, cfg.d_out, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(cfg.d_in, cfg.d_out, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_out, cfg.d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    cfg.context_length,
                    cfg.context_length,
                ),
                diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the multi-head attention.

        :param x: Input tensor of shape (batch_size, num_tokens, d_in).
        :return: Output tensor of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, _d_in = x.shape

        # Shape: (b, num_tokens, d_out)  # noqa: ERA001
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)  # noqa: ERA001
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # contiguous() is used to ensure that the tensor is stored in a contiguous block of memory
        # Caused by the transpose operation, which can lead to non-contiguous memory layout
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        return self.out_proj(context_vec) # optional projection


if __name__ == "__main__":
    torch.manual_seed(123)

    batch_size = 2
    context_length = 6
    d_in, d_out = 8, 16
    num_heads = 4

    cfg = MultiHeadAttentionConfig(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
    )

    mha = MultiHeadAttention(cfg)

    x = torch.randn(batch_size, context_length, d_in)
    out = mha(x)

    print(f"Input shape:  {x.shape}")   # (2, 6, 8)
    print(f"Output shape: {out.shape}") # (2, 6, 16)
    print("Done!")
