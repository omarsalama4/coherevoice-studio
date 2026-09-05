#!/usr/bin/env python3
"""
CohereX LLM package.
Provides LLM-powered meeting intelligence and translation services.

Engine Priority:
1. 🖥️ Local Open-Source LLM (Ollama: Qwen 2.5 7B / 14B) — 100% offline & private on GPU
2. ☁️ Cloud LLM (Google Gemini 2.0 Flash) — If configured via API key
3. ⚡ Heuristic Fallback — If no LLM available
"""

import os
import logging
from typing import Optional

from coherex.llm.client import LLMClient
from coherex.llm.schemas import MeetingMinutes, ActionItem, Decision, TopicDiscussion

logger = logging.getLogger("coherex.llm")

# Singleton client instances
_local_client: Optional[LLMClient] = None
_cloud_client: Optional[LLMClient] = None


def get_local_llm_client(force_new: bool = False) -> Optional[LLMClient]:
    """Get or create the local Ollama LLM client."""
    global _local_client
    if _local_client is not None and not force_new:
        return _local_client

    try:
        from coherex.llm.ollama_client import OllamaClient
        client = OllamaClient()
        if client.is_available():
            _local_client = client
            logger.info("Local LLM client initialized: Ollama [%s]", client.get_active_model())
            return _local_client
    except Exception as e:
        logger.debug("Local LLM not available: %s", e)

    return None


def get_cloud_llm_client(api_key: Optional[str] = None, force_new: bool = False) -> Optional[LLMClient]:
    """Get or create the cloud Gemini LLM client."""
    global _cloud_client
    if _cloud_client is not None and not force_new:
        return _cloud_client

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None

    try:
        from coherex.llm.gemini_client import GeminiClient
        client = GeminiClient(api_key=key)
        if client.is_available():
            _cloud_client = client
            logger.info("Cloud LLM client initialized: Gemini 2.0 Flash")
            return _cloud_client
    except Exception as e:
        logger.debug("Cloud LLM not available: %s", e)

    return None


def get_llm_client(
    prefer_local: bool = True,
    api_key: Optional[str] = None,
    force_new: bool = False
) -> Optional[LLMClient]:
    """
    Get the best available LLM client.
    
    Priority:
    1. Local Ollama LLM (if prefer_local is True and Ollama is running)
    2. Cloud Gemini LLM (if API key is present)
    3. Local Ollama LLM (as secondary if prefer_local was False)
    4. None (triggers rule-based heuristic fallback)
    """
    if prefer_local:
        local = get_local_llm_client(force_new=force_new)
        if local is not None:
            return local

        cloud = get_cloud_llm_client(api_key=api_key, force_new=force_new)
        if cloud is not None:
            return cloud
    else:
        cloud = get_cloud_llm_client(api_key=api_key, force_new=force_new)
        if cloud is not None:
            return cloud

        local = get_local_llm_client(force_new=force_new)
        if local is not None:
            return local

    return None


def is_llm_available(api_key: Optional[str] = None) -> bool:
    """Check if any LLM backend (local or cloud) is available."""
    client = get_llm_client(api_key=api_key)
    return client is not None and client.is_available()


def get_llm_backend_info(api_key: Optional[str] = None) -> dict:
    """Returns detailed status information about available LLM backends."""
    local_client = get_local_llm_client()
    has_local = local_client is not None and local_client.is_available()
    local_model = local_client.get_active_model() if has_local else None

    cloud_client = get_cloud_llm_client(api_key=api_key)
    has_cloud = cloud_client is not None and cloud_client.is_available()

    if has_local:
        active_backend = f"Local ({local_model})"
    elif has_cloud:
        active_backend = "Cloud (Gemini 2.0 Flash)"
    else:
        active_backend = "Heuristic (Rule-Based)"

    return {
        "has_local": has_local,
        "local_model": local_model,
        "has_cloud": has_cloud,
        "active_backend": active_backend,
    }


__all__ = [
    "get_llm_client",
    "get_local_llm_client",
    "get_cloud_llm_client",
    "is_llm_available",
    "get_llm_backend_info",
    "LLMClient",
    "MeetingMinutes",
    "ActionItem",
    "Decision",
    "TopicDiscussion",
]
