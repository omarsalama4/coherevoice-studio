#!/usr/bin/env python3
"""API-backed LLM routing for OpenAI, Groq, and Google Gemini."""

import logging
import os
from typing import Any, Dict, Optional

from coherex.llm.client import LLMClient
from coherex.llm.schemas import ActionItem, Decision, MeetingMinutes, TopicDiscussion

logger = logging.getLogger("coherex.llm")

SUPPORTED_PROVIDERS = ("openai", "groq", "gemini")

_openai_client: Optional[LLMClient] = None
_groq_client: Optional[LLMClient] = None
_gemini_client: Optional[LLMClient] = None


def get_openai_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    force_new: bool = False,
) -> Optional[LLMClient]:
    """Get an OpenAI client when a key is configured."""
    global _openai_client
    if _openai_client is not None and not force_new and not api_key and not model and not base_url:
        return _openai_client

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None

    try:
        from coherex.llm.openai_client import OpenAIClient

        kwargs: Dict[str, Any] = {
            "api_key": key,
            "model": model or os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"),
        }
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAIClient(**kwargs)
        if client.is_available():
            _openai_client = client
            return client
    except Exception as exc:
        logger.debug("OpenAI client unavailable: %s", exc)
    return None


def get_groq_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    force_new: bool = False,
) -> Optional[LLMClient]:
    """Get a GroqCloud client when a key is configured."""
    global _groq_client
    if _groq_client is not None and not force_new and not api_key and not model:
        return _groq_client

    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None

    try:
        from coherex.llm.groq_client import GroqClient

        client = GroqClient(
            api_key=key,
            model=model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        )
        if client.is_available():
            _groq_client = client
            return client
    except Exception as exc:
        logger.debug("Groq client unavailable: %s", exc)
    return None


def get_gemini_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    force_new: bool = False,
) -> Optional[LLMClient]:
    """Get a Google Gemini client when a key is configured."""
    global _gemini_client
    if _gemini_client is not None and not force_new and not api_key and not model:
        return _gemini_client

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None

    try:
        from coherex.llm.gemini_client import GeminiClient

        client = GeminiClient(
            api_key=key,
            model=model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        )
        if client.is_available():
            _gemini_client = client
            return client
    except Exception as exc:
        logger.debug("Gemini client unavailable: %s", exc)
    return None


def _resolve_provider(provider: Optional[str], keys: Dict[str, str]) -> Optional[str]:
    configured = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if configured:
        if configured not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider {configured!r}; choose one of {', '.join(SUPPORTED_PROVIDERS)}"
            )
        return configured
    return next((name for name in SUPPORTED_PROVIDERS if keys.get(name)), None)


def get_llm_client(
    provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    groq_model: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    force_new: bool = False,
) -> Optional[LLMClient]:
    """Return only the selected provider, or the first configured API provider."""
    keys = {
        "openai": openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
        "groq": groq_api_key or os.environ.get("GROQ_API_KEY", ""),
        "gemini": gemini_api_key or os.environ.get("GEMINI_API_KEY", ""),
    }
    selected = _resolve_provider(provider, keys)
    if selected == "openai":
        return get_openai_client(keys["openai"], openai_model, openai_base_url, force_new)
    if selected == "groq":
        return get_groq_client(keys["groq"], groq_model, force_new)
    if selected == "gemini":
        return get_gemini_client(keys["gemini"], gemini_model, force_new)
    return None


def is_llm_available(**kwargs: Any) -> bool:
    """Check whether the selected API provider is configured."""
    client = get_llm_client(**kwargs)
    return client is not None and client.is_available()


def get_llm_backend_info(
    provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Return non-network status for configured API providers."""
    keys = {
        "openai": openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
        "groq": groq_api_key or os.environ.get("GROQ_API_KEY", ""),
        "gemini": gemini_api_key or os.environ.get("GEMINI_API_KEY", ""),
    }
    selected = _resolve_provider(provider, keys)
    return {
        "selected_provider": selected,
        "has_openai": bool(keys["openai"]),
        "has_groq": bool(keys["groq"]),
        "has_gemini": bool(keys["gemini"]),
        "any_available": any(keys.values()),
    }


__all__ = [
    "SUPPORTED_PROVIDERS",
    "get_llm_client",
    "get_openai_client",
    "get_groq_client",
    "get_gemini_client",
    "is_llm_available",
    "get_llm_backend_info",
    "LLMClient",
    "MeetingMinutes",
    "ActionItem",
    "Decision",
    "TopicDiscussion",
]
