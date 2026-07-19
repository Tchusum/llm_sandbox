import tiktoken
from pydantic import BaseModel, ConfigDict, Field
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer

from llm_sandbox.llm.models import LLMGPTModel, LLMQwen3Model


class LLMModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Qwen3Tokenizer | tiktoken.Encoding = Field(..., description="The tokenizer used for the LLM model.")
    model: LLMGPTModel | LLMQwen3Model = Field(..., description="The LLM model instance.")
    name: str = Field(..., description="The name of the LLM model.")
    eos_id: int = Field(..., description="The end-of-sequence token ID for the model.")
