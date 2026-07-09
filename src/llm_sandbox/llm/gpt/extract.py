from llm_sandbox.llm.extract import download_json, split_instruction_data

INSTRUCTION_DATASETS = {
    "rasbt": {
        "file_path": "data/gpt2/instruction-data-rasbt.json",
        "url": (
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
            "/main/ch07/01_main-chapter-code/instruction-data.json"
        ),
    },
    "alpaca": {
        "file_path": "data/gpt2/instruction-data-alpaca.json",
        "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    },
}

def query_instruction_data(
    dataset_name: str = "rasbt",
    train_ratio: float = 0.85,
    test_ratio: float = 0.1,
    max_samples: int | None = None,
) -> tuple[list, list, list]:
    """Load a supported instruction dataset and split it for fine-tuning."""
    try:
        dataset_config = INSTRUCTION_DATASETS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(INSTRUCTION_DATASETS))
        msg = f"Unsupported dataset '{dataset_name}'. Choose one of: {supported}."
        raise ValueError(msg) from exc

    data = download_json(dataset_config["file_path"], dataset_config["url"])
    if max_samples is not None:
        data = data[:max_samples]
    return split_instruction_data(data, train_ratio=train_ratio, test_ratio=test_ratio)
