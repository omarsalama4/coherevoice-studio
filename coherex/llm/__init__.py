#!/usr/bin/env python3
"""
CohereX LLM package.
Provides LLM-powered meeting intelligence and translation services.
"""

import os
import logging
from typing import Optional

from coherex.llm.client import LLMClient
from coherex.llm.schemas import MeetingMinutes, ActionItem, Decision, TopicDiscussion

logger = logging.getLogger("coherex.llm")

# Singleton client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(api_key: Optional[str] = None, force_new: bool = False) -> Optional[LLMClient]:
    """
    Get or create the singleton LLM client.
    
    Attempts to initialize a Gemini client using (in order):
    1. The provided api_key argument
    2. The GEMINI_API_KEY environment variable
    
    Returns None if no API key is available.
    
    Args:
        api_key: Optional explicit API key. If provided, overrides environment.
        force_new: If True, creates a new client even if one already exists.
        
    Returns:
        LLMClient instance or None if no API key configured.
    """
    global _llm_client

    if _llm_client is not None and not force_new:
        return _llm_client

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        logger.info("No GEMINI_API_KEY found — LLM features disabled (heuristic fallback active)")
        return None

    try:
        from coherex.llm.gemini_client import GeminiClient
        client = GeminiClient(api_key=key)
        if client.is_available():
            _llm_client = client
            logger.info("LLM client initialized: Gemini 2.0 Flash")
            return _llm_client
        else:
            logger.warning("Gemini client created but not available")
            return None
    except Exception as e:
        logger.warning("Failed to create LLM client: %s", e)
        return None


def is_llm_available(api_key: Optional[str] = None) -> bool:
    """Quick check if an LLM backend is configured and reachable."""
    client = get_llm_client(api_key=api_key)
    return client is not None and client.is_available()


__all__ = [
    "get_llm_client",
    "is_llm_available",
    "LLMClient",
    "MeetingMinutes",
    "ActionItem",
    "Decision",
    "TopicDiscussion",
]
