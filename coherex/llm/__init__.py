#!/usr/bin/env python3
"""
CohereX LLM package.
Multi-backend LLM intelligence supporting both Meetings and Videos.

Supported Backends:
1. 🖥️ Local Open-Source LLM (Ollama: Qwen 2.5 7B / 14B) — 100% offline & private on GPU
2. 🤖 OpenAI API (GPT-4o, GPT-4o-mini, or any OpenAI-compatible endpoint like Groq, vLLM)
3. ☁️ Google Gemini API (Gemini 2.0 Flash)
4. ⚡ Heuristic Rule-Based Fallback
"""

import os
import logging
from typing import Optional, Dict, Any, List

from coherex.llm.client import LLMClient
from coherex.llm.schemas import MeetingMinutes, ActionItem, Decision, TopicDiscussion

logger = logging.getLogger("coherex.llm")

# Singleton client caches
_local_client: Optional[LLMClient] = None
_openai_client: Optional[LLMClient] = None
_gemini_client: Optional[LLMClient] = None


def get_local_llm_client(auto_start: bool = True, force_new: bool = False) -> Optional[LLMClient]:
    """Get or create the local Ollama LLM client with auto-start capability."""
    global _local_client
    if _local_client is not None and not force_new:
        return _local_client

    try:
        from coherex.llm.ollama_client import OllamaClient
        client = OllamaClient(auto_start=auto_start)
        if client.is_available():
            _local_client = client
            logger.info("Local LLM online: Ollama [%s]", client.get_active_model())
            return _local_client
    except Exception as e:
        logger.debug("Local LLM not available: %s", e)

    return None


def get_openai_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    force_new: bool = False
) -> Optional[LLMClient]:
    """Get or create the OpenAI-compatible LLM client."""
    global _openai_client
    if _openai_client is not None and not force_new and not api_key:
        return _openai_client

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None

    try:
        from coherex.llm.openai_client import OpenAIClient
        kwargs: Dict[str, Any] = {"api_key": key}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAIClient(**kwargs)
        if client.is_available():
            _openai_client = client
            logger.info("OpenAI client initialized: [%s]", client.model)
            return _openai_client
    except Exception as e:
        logger.debug("OpenAI client not available: %s", e)

    return None


def get_gemini_client(api_key: Optional[str] = None, force_new: bool = False) -> Optional[LLMClient]:
    """Get or create the cloud Gemini LLM client."""
    global _gemini_client
    if _gemini_client is not None and not force_new and not api_key:
        return _gemini_client

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None

    try:
        from coherex.llm.gemini_client import GeminiClient
        client = GeminiClient(api_key=key)
        if client.is_available():
            _gemini_client = client
            logger.info("Gemini client initialized: Gemini 2.0 Flash")
            return _gemini_client
    except Exception as e:
        logger.debug("Gemini client not available: %s", e)

    return None


def get_llm_client(
    prefer_local: bool = True,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    force_new: bool = False,
) -> Optional[LLMClient]:
    """
    Get the best available LLM client across all backends.
    
    Priority:
    1. Local Ollama LLM (auto-started if needed, running 100% offline)
    2. OpenAI API (if configured via key)
    3. Google Gemini API (if configured via key)
    4. None (triggers rule-based heuristic fallback)
    """
    if prefer_local:
        local = get_local_llm_client(auto_start=True, force_new=force_new)
        if local is not None and local.is_available():
            return local

    # Check OpenAI
    openai_client = get_openai_client(
        api_key=openai_api_key,
        model=openai_model,
        base_url=openai_base_url,
        force_new=force_new
    )
    if openai_client is not None and openai_client.is_available():
        return openai_client

    # Check Gemini
    gemini = get_gemini_client(api_key=gemini_api_key, force_new=force_new)
    if gemini is not None and gemini.is_available():
        return gemini

    # If prefer_local was False, still try local as last resort
    if not prefer_local:
        local = get_local_llm_client(auto_start=True, force_new=force_new)
        if local is not None and local.is_available():
            return local

    return None


def is_llm_available(
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> bool:
    """Check if any LLM backend (local, OpenAI, or Gemini) is available."""
    client = get_llm_client(gemini_api_key=gemini_api_key, openai_api_key=openai_api_key)
    return client is not None and client.is_available()


def get_llm_backend_info(
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns fast, non-blocking status information about all available LLM backends.
    Used by the Streamlit UI to render real-time status without freezing.
    """
    local_client = get_local_llm_client(auto_start=False)
    has_local = local_client is not None and local_client.is_available()
    local_model = local_client.get_active_model() if has_local else None
    installed_models = local_client.get_installed_models() if has_local else []

    openai_client = get_openai_client(api_key=openai_api_key)
    has_openai = openai_client is not None and openai_client.is_available()
    openai_model = openai_client.model if has_openai else None

    gemini_client = get_gemini_client(api_key=gemini_api_key)
    has_gemini = gemini_client is not None and gemini_client.is_available()

    if has_local:
        active_backend = f"Local Ollama ({local_model})"
        backend_type = "local"
    elif has_openai:
        active_backend = f"OpenAI ({openai_model})"
        backend_type = "openai"
    elif has_gemini:
        active_backend = "Google Gemini 2.0"
        backend_type = "gemini"
    else:
        active_backend = "Rule-Based Heuristic"
        backend_type = "heuristic"

    return {
        "has_local": has_local,
        "local_model": local_model,
        "installed_models": installed_models,
        "has_openai": has_openai,
        "openai_model": openai_model,
        "has_gemini": has_gemini,
        "active_backend": active_backend,
        "backend_type": backend_type,
        "any_available": has_local or has_openai or has_gemini,
    }


__all__ = [
    "get_llm_client",
    "get_local_llm_client",
    "get_openai_client",
    "get_gemini_client",
    "is_llm_available",
    "get_llm_backend_info",
    "LLMClient",
    "MeetingMinutes",
    "ActionItem",
    "Decision",
    "TopicDiscussion",
]
