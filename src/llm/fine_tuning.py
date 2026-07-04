"""Fine-tuning utilities for large language models (LLMs)."""
import json
import pathlib
import re
import time
from functools import partial

import psutil
import requests
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llm.extract import query_instruction_data
from llm.model import generate
from llm.tokenizer import text_to_token_ids, token_ids_to_text
from llm.training import load_gpt2_model, train_model_simple


def format_input(entry: dict) -> str:
    """Format the instruction and input fields of a dataset entry into a single string.

    :param entry: A dictionary containing 'instruction' and 'input' keys.
    :return: A formatted string combining the instruction and input.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    """Custom dataset for instruction-based fine-tuning of LLMs."""

    def __init__(self, data: list, tokenizer: tiktoken.Encoding) -> None:
        """Set up the dataset with pre-tokenized instruction data.

        :param data: A list of dictionaries, each containing 'instruction', 'input', and 'output' keys.
        :param tokenizer: A tiktoken.Encoding instance for tokenizing the text data.
        """
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text),
            )

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrieve the tokenized text at the specified index.

        :param index: The index of the data point to retrieve.
        :return: A torch.Tensor containing the token IDs for the specified data point.
        """
        return self.encoded_texts[index]

    def __len__(self) -> int:
        """Return the total number of data points in the dataset.

        :return: The length of the dataset.
        """
        return len(self.data)


def custom_collate_fn(
    batch: list,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare batches of tokenized sequences for model training.

    :param batch: A list of tokenized sequences to be collated.
    :param pad_token_id: The token ID used for padding sequences.
    :param ignore_index: The index to use for ignored tokens in the target sequences.
    :param allowed_max_length: Optional maximum length for sequences; if provided,
        sequences will be truncated to this length.
    :param device: The device to which the resulting tensors will be moved.
    :return: A tuple containing the input tensor and target tensor, both moved to
        the specified device.
    """
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def create_data_loader(  # noqa: PLR0913
    data: list,
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    device: str = "cpu",
    allowed_max_length: int = 1024,
) -> DataLoader:
    """Create a DataLoader for instruction dataset.

    :param data: Dataset list
    :param tokenizer: tiktoken encoding
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param shuffle: Whether to shuffle data
    :param drop_last: Whether to drop last incomplete batch
    :param device: Device to move data to
    :param allowed_max_length: Max sequence length
    :return: Configured DataLoader
    """
    dataset = InstructionDataset(data, tokenizer)
    collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=allowed_max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def check_if_running(process_name: str) -> bool:
    """Check if a process with the given name is currently running.

    :param process_name: The name of the process to check.
    :return: True if the process is running, False otherwise.
    """
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def query_model(
    prompt: str,
    model: str = "llama3",
    url: str = "http://localhost:11434/api/chat",
) -> str:
    """Query a language model via a REST API and return the response.

    :param prompt: The input prompt to send to the model.
    :param model: The name of the model to query.
    :param url: The URL of the REST API endpoint for querying the model.
    :return: The model's response as a string.
    """
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048,
        },
    }

    # Send the POST request
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data


def generate_model_scores(
    json_data: list,
    json_key: str,
    model: str = "llama3",
) -> list[int]:
    """Generate scores for model responses based on a given dataset.

    :param json_data: A list of dictionaries, each containing 'instruction', 'input', 'output', and model response keys.
    :param json_key: The key in the dictionaries that corresponds to the model's response to be scored.
    :param model: The name of the model to query for scoring.
    :return: A list of integer scores corresponding to the model responses.
    """
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores

def gpt2_fine_tuning_wrapper(  # noqa: PLR0913
    model_name: str,
    dataset_name: str,
    batch_size: int = 8,
    num_workers: int = 0,
    num_epochs: int = 2,
    allowed_max_length: int = 1024,
    accumulation_steps: int = 1,
    max_samples: int | None = None,
    *,
    evaluate_with_ollama: bool = True,
) -> None:
    """Fine-tune GPT-2 on a supported instruction dataset.

    :param model_name: GPT-2 model name to load.
    :param dataset_name: Instruction dataset name. Supported values include "rasbt" and "alpaca".
    :param batch_size: Training and validation batch size.
    :param num_workers: Number of DataLoader workers.
    :param num_epochs: Number of training epochs.
    :param allowed_max_length: Maximum token length passed to the collate function.
        :param accumulation_steps: Number of batches to accumulate gradients over. Use 4 with batch_size=2
                                to simulate batch_size=8 with less memory.
    :param max_samples: Optional maximum number of samples to use from the dataset.
    :param evaluate_with_ollama: Whether to score test responses through the local Ollama API.
    """
    # Download data
    train_data, val_data, test_data = query_instruction_data(dataset_name, max_samples=max_samples)

    # Set up data loaders
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader = create_data_loader(
        train_data,
        tokenizer,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=True,
        device=device,
        allowed_max_length=allowed_max_length,
    )
    val_loader = create_data_loader(
        val_data,
        tokenizer,
        batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        device=device,
        allowed_max_length=allowed_max_length,
    )

    # Load a GPT-2 model
    model, param = load_gpt2_model(model_name)

    # Fine-tune the model on the instruction dataset
    start_time = time.time()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
        accumulation_steps=accumulation_steps,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Generate model responses for the test dataset
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=param.context_length,
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    output_path = pathlib.Path("data")
    response_path = output_path / f"instruction-data-{dataset_name}-with-response.json"
    with response_path.open("w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

    normalized_model_name = re.sub(r"[ ()]", "", model_name)
    file_name = f"{normalized_model_name}-{dataset_name}-sft.pth"
    torch.save(model.state_dict(), output_path / file_name)
    print(f"Model saved as {file_name}")

    # Evaluate the model responses using the scoring function
    if not evaluate_with_ollama:
        return

    ollama_running = check_if_running("ollama")

    if not ollama_running:
        msg = "Ollama not running. Launch ollama before proceeding."
        raise RuntimeError(msg)
    print("Ollama running:", check_if_running("ollama"))

    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


if __name__ == "__main__":

    torch.manual_seed(123)
    model_name = "gpt2-xl"
    gpt2_fine_tuning_wrapper(
        model_name,
        dataset_name="alpaca",
        batch_size=1,
        max_samples=500,
    )

