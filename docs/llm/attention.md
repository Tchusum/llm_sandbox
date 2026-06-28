# Multi-Head Self-Attention

This module implements the **Multi-Head Self-Attention (MHA)** layer used in Transformer and GPT models. It allows each token to gather information from the most relevant previous tokens.

## Pipeline

```
Input
  │
  ▼
Linear Layers (Q, K, V)
  │
  ▼
Split into Multiple Heads
  │
  ▼
Scaled Dot-Product Attention
  │
  ▼
Concatenate Heads
  │
  ▼
Output Projection
```

## Shape Flow

Assume:

- `batch_size = 2`
- `num_tokens = 6`
- `d_in = 8`
- `d_out = 16`
- `num_heads = 4`
- `head_dim = 4`

| Step | Shape |
|------|-------|
| Input | `(2, 6, 8)` |
| Q, K, V | `(2, 6, 16)` |
| Split Heads | `(2, 6, 4, 4)` |
| Transpose | `(2, 4, 6, 4)` |
| Attention Scores | `(2, 4, 6, 6)` |
| Context Vectors | `(2, 4, 6, 4)` |
| Concatenate Heads | `(2, 6, 16)` |
| Output Projection | `(2, 6, 16)` |

## Main Steps

1. **Compute Q, K, V**
   - Project each input embedding into Query, Key, and Value vectors.

2. **Split into heads**
   - Divide the embedding into `num_heads` smaller representations.

3. **Compute attention**
   - Calculate `Q × Kᵀ` for each head.

4. **Apply causal mask**
   - Prevent each token from attending to future tokens.

5. **Scale + Softmax**
   - Divide by `√head_dim`, then convert scores into attention probabilities.

6. **Weighted sum**
   - Multiply attention weights by the Value vectors to obtain context vectors.

7. **Merge heads**
   - Concatenate all head outputs back into a single embedding.

8. **Output projection**
   - Mix information from all heads using a final linear layer.

## Why Multiple Heads?

Each head can learn a different relationship, such as:

- Local context
- Long-range dependencies
- Syntax
- Punctuation

The final output combines information from all heads into one contextualized embedding.

## Final Output

The layer returns:

```text
(batch_size, num_tokens, d_out)
```

Each output token is an enriched representation that contains information from the most relevant previous tokens.