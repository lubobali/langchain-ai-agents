"""Shared test infrastructure — real API calls."""

import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@pytest.fixture
def llm():
    """ChatOpenAI pointed at OpenAI-compatible API. Fresh session per test."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL"),  # Optional: set for custom proxy
        streaming=True,
    )
