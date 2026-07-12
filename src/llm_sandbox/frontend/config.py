from llm_sandbox.llm.gpt.config import get_model_schema_gpt
from llm_sandbox.llm.qwen3.config import get_model_schema_qwen3

LLM_CONFIGS = {
    "gpt2": get_model_schema_gpt,
    "qwen3": get_model_schema_qwen3,
}
