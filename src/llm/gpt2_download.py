"""Module for downloading and loading GPT-2 models."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from llm.extract import download_file


def download_and_load_gpt2(model_size: str, models_dir: str) -> dict:
    """Download and load GPT-2 model parameters from TensorFlow checkpoints.

    :param model_size: The size of the GPT-2 model to download
    :param models_dir: The directory where the model files will be stored.
    :return: A dictionary containing the model parameters.
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        msg = f"Model size not in {allowed_sizes}"
        raise ValueError(msg)

    # Define paths
    model_dir = Path(models_dir) / model_size
    # https://github.com/openai/gpt-2/blob/master/download_model.py
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe",
    ]

    # Download files
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        file_path = Path(model_dir) / filename
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    with (Path(model_dir) / "hparams.json").open("r", encoding="utf-8") as f:
        settings = json.load(f)
    return load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict) -> dict:
    """Load GPT-2 model parameters from a TensorFlow checkpoint.

    :param ckpt_path: The path to the TensorFlow checkpoint.
    :param settings: The model settings dictionary.
    :return: A dictionary containing the model parameters.
    """
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


if __name__ == "__main__":
    model_size = "124M"
    models_dir = "data/gpt2"
    params = download_and_load_gpt2(model_size, models_dir)
    print(f"Loaded GPT-2 model parameters for size {model_size}.")
