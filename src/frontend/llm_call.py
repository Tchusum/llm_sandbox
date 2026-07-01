"""Usage example of the LLM model call."""
import tiktoken
import torch

from llm.model import generate, load_model
from llm.tokenizer import text_to_token_ids, token_ids_to_text

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, config = load_model("gpt2-xl", device=device)

prompt = """
### Instruction:
Whish me a good day.

### Input:
The horse is fast.
"""
tokenizer = tiktoken.get_encoding("gpt2")
prompt_token = text_to_token_ids(prompt, tokenizer).to(device)

response_token = generate(
    model = model,
    idx = prompt_token,
    max_new_tokens = 50,
    context_size = config.context_length,
    eos_id = 50256,
)
response = token_ids_to_text(response_token, tokenizer)
print(response)
