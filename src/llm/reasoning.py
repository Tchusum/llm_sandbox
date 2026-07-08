from pathlib import Path

import torch
from reasoning_from_scratch.qwen3 import QWEN_CONFIG_06_B, Qwen3Model, Qwen3Tokenizer, download_qwen3_small

from llm.model import generate
from llm.tokenizer import token_ids_to_text
from llm.utils import get_device

OUT_DIR = Path("data/qwen3")

download_qwen3_small(kind="base", tokenizer_only=False, out_dir=OUT_DIR)

tokenizer_path = OUT_DIR / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

device = get_device()

model_path = OUT_DIR / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Test
prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
    ).unsqueeze(0)
max_new_tokens = 100

output_token = generate(
    model=model,
    idx=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    context_size=512,
    exclude_input=True,
)
output_text = token_ids_to_text(output_token, tokenizer)
print(output_text)

