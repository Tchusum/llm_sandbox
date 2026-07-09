import logging

import torch

logger = logging.getLogger(__name__)

def get_device() -> torch.device:

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
