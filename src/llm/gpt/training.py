"""Load pretrained GPT-2 model, assign weights, and generate text."""
import numpy as np
import tiktoken
import torch

from llm.gpt.config import MODEL_CONFIG
from llm.gpt.download import download_and_load_gpt2
from llm.models import GPTConfig, GPTModel, generate
from llm.tokenizer import text_to_token_ids, token_ids_to_text
from llm.utils import get_device


def load_gpt2_model(model_name: str) -> tuple[GPTModel, GPTConfig]:
    """Load a GPT-2 model with pretrained weights from TensorFlow checkpoints.

    :param model_name: The name of the GPT-2 model to load (e.g., "gpt2-small (124M)").
    :return: A tuple containing the GPT model instance and its configuration.
    """
    # Copy the base configuration and update with specific model settings
    config = GPTConfig(
        emb_dim=MODEL_CONFIG[model_name]["emb_dim"],
        n_layers=MODEL_CONFIG[model_name]["n_layers"],
        n_heads=MODEL_CONFIG[model_name]["n_heads"],
        qkv_bias=True,
    )

    params = download_and_load_gpt2(model_size=MODEL_CONFIG[model_name]["size"], models_dir="data/gpt2")

    gpt = GPTModel(config)
    gpt.eval()

    load_weights_into_gpt(gpt, params)

    device = get_device()
    gpt.to(device)

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


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Calculate the loss for a batch of input and target data.

    :param input_batch: A batch of input data (token IDs).
    :param target_batch: A batch of target data (token IDs).
    :param model: The GPT model used for generating predictions.
    :param device: The device (CPU or GPU) to run the calculations on.
    :return: The calculated loss for the batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Calculate the average loss over a specified number of batches from a data loader.

    :param data_loader: A DataLoader providing batches of input and target data.
    :param model: The GPT model used for generating predictions.
    :param device: The device (CPU or GPU) to run the calculations on.
    :param num_batches: The number of batches to use for loss calculation.
    :return: The average loss over the specified number of batches.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: GPTModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Evaluate the model on training and validation data.

    :param model: The GPT model to evaluate.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param device: The device (CPU or GPU) to run the evaluation on.
    :param eval_iter: Number of iterations to use for evaluation.
    :return: A tuple containing the average training loss and validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
) -> None:
    """Generate a sample text from the model and print it.

    :param model: The GPT model used for generating text.
    :param tokenizer: The tokenizer used for encoding and decoding text.
    :param device: The device (CPU or GPU) to run the generation on.
    :param start_context: The initial context string for generating sample text.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(  # noqa: PLR0913
    model: GPTModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding,
    accumulation_steps: int = 1,
) -> tuple[list[float], list[float], list[int]]:
    """Train the model using a simple training loop.

    :param model: The GPT model to be trained.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param optimizer: Optimizer for updating model weights.
    :param device: The device (CPU or GPU) to run the training on.
    :param num_epochs: Number of epochs to train the model.
    :param eval_freq: Frequency (in steps) to evaluate the model on validation data.
    :param eval_iter: Number of iterations to use for evaluation.
    :param start_context: The initial context string for generating sample text.
    :param tokenizer: The tokenizer used for encoding and decoding text.
    :param accumulation_steps: Number of batches to accumulate gradients over.
        It use less memory and can be used to simulate larger batch sizes.
    :return: A tuple containing lists of training losses, validation losses, and tokens seen.
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Forward pass with gradient accumulation
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss = loss / accumulation_steps  # Scale loss for gradient averaging
            loss.backward()  # Accumulate gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Update weights every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context,
        )

    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    torch.manual_seed(123)

    gpt, config = load_gpt2_model("gpt2-small")

    tokenizer = tiktoken.get_encoding("gpt2")

    device = get_device()
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=config.context_length,
        top_k=50,
        temperature=1.5,
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
