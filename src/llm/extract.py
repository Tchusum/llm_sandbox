import json
from pathlib import Path

import requests
from tqdm import tqdm


def download_json(file_path: str, url: str) -> dict:
    """Download a JSON file from a URL and save it locally if it doesn't exist.

    :param file_path: The local path where the JSON file will be saved.
    :param url: The URL of the JSON file to download.
    :return: The loaded JSON data as a dictionary.
    """
    if not Path(file_path).exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with Path(file_path).open("w", encoding="utf-8") as file:
            file.write(text_data)

    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def download_url_file(url: str, destination: Path) -> None:
    """Download a file from a URL with progress tracking.

    :param url: The URL of the file to download.
    :param destination: The local path where the file will be saved.
    """
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True, timeout=30)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if destination.exists():
        file_size_local = destination.stat().st_size
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.rsplit("/", 1)[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar, \
            destination.open("wb") as file:
        # Iterate over the file data in chunks
        for chunk in response.iter_content(block_size):
            progress_bar.update(len(chunk))
            file.write(chunk)


def split_instruction_data(
    data: list,
    train_ratio: float = 0.85,
    test_ratio: float = 0.1,
) -> tuple[list, list, list]:
    """Split instruction data into train, validation, and test sets."""
    train_portion = int(len(data) * train_ratio)
    test_portion = int(len(data) * test_ratio)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    return train_data, val_data, test_data
