# Model Module Summary

`src/llm/model.py` implements a small GPT-like language model in PyTorch. It defines the model configuration, core transformer building blocks, the full GPT model, and a simple greedy text generation helper.

GPT model
│
├─ Linear output layer
├─ Final LayerNorm
│
├─ Transformer block × 12
│  │
│  ├─ LayerNorm 1
│  ├─ Masked multi-head attention
│  ├─ Dropout
│  ├─ Residual connection
│  │
│  ├─ LayerNorm 2
│  ├─ Feed forward
│  ├─ Dropout
│  └─ Residual connection
│
├─ Dropout
├─ Positional embedding layer
├─ Token embedding layer
└─ Tokenized text

## Configuration

`GPT_CONFIG_124M` stores the default architecture settings for a GPT-2-style 124M parameter model:

- `vocab_size`: number of token IDs in the vocabulary.
- `context_length`: maximum number of tokens the model can attend to.
- `emb_dim`: token and positional embedding size.
- `n_heads`: number of attention heads per transformer block.
- `n_layers`: number of stacked transformer blocks.
- `drop_rate`: dropout probability used in embeddings, attention, and residual paths.
- `qkv_bias`: whether query, key, and value projections include bias terms.

## Components

### `GELU`

`GELU` implements the Gaussian Error Linear Unit activation used in GPT-style feed-forward networks. The implementation uses the common tanh approximation.

### `FeedForward`

`FeedForward` is the per-token multilayer perceptron inside each transformer block. It expands the embedding dimension by a factor of four, applies `GELU`, then projects back to the original embedding dimension.

Input and output shape:

```text
[batch_size, num_tokens, emb_dim] -> [batch_size, num_tokens, emb_dim]
```

### `LayerNorm`

`LayerNorm` normalizes each token representation across the embedding dimension. It keeps learned `scale` and `shift` parameters so the model can adjust the normalized output.

### `TransformerBlock`

`TransformerBlock` combines:

- causal multi-head self-attention from `llm.attention.MultiHeadAttention`;
- a feed-forward network;
- layer normalization before each sublayer;
- dropout on each residual branch;
- residual shortcut connections around both attention and feed-forward paths.

The block uses a pre-normalization layout:

```text
x -> LayerNorm -> Attention -> Dropout -> Residual Add
x -> LayerNorm -> FeedForward -> Dropout -> Residual Add
```

## `GPTModel`

`GPTModel` builds the full language model:

1. Token IDs are converted to token embeddings.
2. Positional embeddings are created for the current sequence length.
3. Token and positional embeddings are added together.
4. Dropout is applied to the combined embeddings.
5. The sequence passes through `n_layers` transformer blocks.
6. A final layer normalization is applied.
7. A bias-free linear output head maps each token representation to vocabulary logits.

Forward input shape:

```text
[batch_size, num_tokens]
```

Forward output shape:

```text
[batch_size, num_tokens, vocab_size]
```

The output logits can be used for next-token prediction during training or inference.

## Text Generation

`generate_text_simple` performs basic autoregressive generation:

1. Keep only the most recent `context_size` tokens.
2. Run the model without gradient tracking.
3. Select logits for the final token position.
4. Convert logits to probabilities with softmax.
5. Choose the highest-probability token with `argmax`.
6. Append that token to the running sequence.

This is deterministic greedy decoding. It does not use sampling, temperature, top-k filtering, or nucleus sampling.
