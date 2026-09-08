#!/usr/bin/env python3
"""GroqCloud LLM client using Groq's OpenAI-compatible API."""

import os
from typing import Optional

from coherex.llm.openai_client import OpenAIClient

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqClient(OpenAIClient):
    """CohereX intelligence client for hosted GroqCloud models."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(
            api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
            model=model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            base_url=DEFAULT_GROQ_BASE_URL,
        )
        self.provider_name = "groq"
