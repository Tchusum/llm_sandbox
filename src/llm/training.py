"""Load pretrained GPT-2 model, assign weights, and generate text."""
import numpy as np
import tiktoken
import torch

from llm.gpt2_download import download_and_load_gpt2
from llm.model import GPTConfig, GPTModel, generate
from llm.tokenizer import text_to_token_ids, token_ids_to_text

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def load_gpt2_model() -> tuple[GPTModel, GPTConfig]:
    """Load a GPT-2 model with pretrained weights from TensorFlow checkpoints.

    :return: A tuple containing the GPT model instance and its configuration.
    """
    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    config = GPTConfig(
        emb_dim=model_configs[model_name]["emb_dim"],
        n_layers=model_configs[model_name]["n_layers"],
        n_heads=model_configs[model_name]["n_heads"],
        qkv_bias=True,
    )

    params = download_and_load_gpt2(model_size="124M", models_dir="data/gpt2")

    gpt = GPTModel(config)
    gpt.eval()

    load_weights_into_gpt(gpt, params)
    gpt.to("cpu")

    return gpt, config


def assign(left: torch.nn.Parameter, right: np.ndarray) -> torch.nn.Parameter:
    """Assign values from a NumPy array to a PyTorch parameter, ensuring shape compatibility.

    :param left: The PyTorch parameter to which values will be assigned.
    :param right: The NumPy array containing the values to assign.
    :return: The updated PyTorch parameter with values from the NumPy array.
    :raises ValueError: If the shapes of the left and right parameters do not match.
    """
    if left.shape != right.shape:
        msg = f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        raise ValueError(msg)
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(
        gpt: GPTModel,
        params: dict,
) -> None:
    """Load pretrained weights from the params dictionary into the GPT model.

    :param gpt: The GPT model instance into which weights will be loaded.
    :param params: A dictionary containing pretrained weights for the model.
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

if __name__ == "__main__":
    torch.manual_seed(123)

    gpt, config = load_gpt2_model()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to("cpu"),
        max_new_tokens=25,
        context_size=config.context_length,
        top_k=50,
        temperature=1.5,
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
