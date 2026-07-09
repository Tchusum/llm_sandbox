"""Streamlit app."""
import streamlit as st
import tiktoken
import torch

from llm_sandbox.frontend.llm_call import llm_call
from llm_sandbox.llm.models import GPTConfig, GPTModel, load_model
from llm_sandbox.llm.utils import get_device


@st.cache_resource
def get_device_st() -> torch.device:
    return get_device()


@st.cache_resource
def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")

@st.cache_resource
def get_model_and_config(device: torch.device) -> tuple[GPTModel, GPTConfig]:
    return load_model("gpt2-xl-alpaca-sft", device=device)


def app() -> None:
    st.title("LLM Sandbox")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    device = get_device_st()
    tokenizer = get_tokenizer()
    model, config = get_model_and_config(device)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a simple question?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = llm_call(prompt, model, config, tokenizer, device)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    app()
# uv run --active streamlit run src/frontend/app.py
