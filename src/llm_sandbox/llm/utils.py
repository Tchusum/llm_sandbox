from __future__ import annotations

import logging

import torch


def get_device() -> torch.device:

    logger = get_logger()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using NVIDIA CUDA GPU")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        logger.info("Using Intel GPU")

    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def get_logger(name: str = __name__) -> logging.Logger:

    # Configure the logging system
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Create a logger for this module
    logger = logging.getLogger(name)

    return logger
