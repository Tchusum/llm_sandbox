"""Utility functions."""
import json
import pathlib

import requests


def download_json(file_path: str, url: str) -> dict:
    """Download a JSON file from a URL and save it locally if it doesn't exist.

    :param file_path: The local path where the JSON file will be saved.
    :param url: The URL of the JSON file to download.
    :return: The loaded JSON data as a dictionary.
    """
    if not pathlib.Path(file_path).exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with pathlib.Path(file_path).open("w", encoding="utf-8") as file:
            file.write(text_data)

    with pathlib.Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)
