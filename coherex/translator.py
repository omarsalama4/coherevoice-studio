#!/usr/bin/env python3
"""
General-Purpose Dynamic Subtitle & Transcript Translator for CohereX.

Translation is performed only by a configured LLM provider.

1. LLM-Powered: Context-aware, dialect-savvy batch translation that
   understands Egyptian Arabic idioms, code-switching, and conversational tone.
"""

import re
import sys
import copy
import logging
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger("coherex.translator")

POPULAR_LANGUAGES = {
    "English": "en",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Turkish": "tr",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Korean": "ko",
    "Hindi": "hi",
    "Dutch": "nl",
    "Polish": "pl",
    "Greek": "el",
    "Swedish": "sv",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Czech": "cs",
    "Romanian": "ro",
    "Ukrainian": "uk",
    "Hebrew": "iw",
    "Persian (Farsi)": "fa",
    "Urdu": "ur"
}


# ==============================================================================
# TEXT CLEANING UTILITIES
# ==============================================================================

def clean_asr_repetitions(text: str) -> str:
    """Removes degenerate ASR looping patterns caused by background music or noise."""
    if not text:
        return ""
    # Remove character repeats (e.g., Faaaaaaaaaaaa Aaaaaaaaaaaa)
    text = re.sub(r'([a-zA-Z\u0600-\u06FF])\1{3,}', r'\1', text)
    text = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'(\b\w+\s+\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def polish_translated_text(text: str) -> str:
    """Normalizes subtitle punctuation, spacing, and capitalization."""
    if not text:
        return ""
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def validate_translation_batch(
    source_texts: List[str], translated_texts: List[str], target_lang: str
) -> None:
    """Reject incomplete or obviously untranslated subtitle batches."""
    if len(translated_texts) != len(source_texts):
        raise TranslationUnavailableError(
            f"Translation count mismatch: expected {len(source_texts)}, got {len(translated_texts)}"
        )
    for index, (source, translated) in enumerate(zip(source_texts, translated_texts), 1):
        if not isinstance(translated, str) or (source.strip() and not translated.strip()):
            raise TranslationUnavailableError(f"Translation {index} is empty or not text")
        if target_lang.lower().startswith("en") and re.search(r"[\u0600-\u06ff]", source):
            if translated.strip() == source.strip():
                raise TranslationUnavailableError(f"Translation {index} copied the Arabic source")
            arabic_chars = len(re.findall(r"[\u0600-\u06ff]", translated))
            letters = len(re.findall(r"[A-Za-z\u0600-\u06ff]", translated))
            if letters and arabic_chars / letters > 0.10:
                raise TranslationUnavailableError(
                    f"Translation {index} contains excessive Arabic script for an English track"
                )


# ==============================================================================
# DISABLED LEGACY FALLBACK COMPATIBILITY API
# ==============================================================================

class TranslationUnavailableError(RuntimeError):
    """Raised when no configured, trustworthy translation provider is available."""


def raw_neural_translate(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Compatibility entry point retained for callers. Public unofficial translation
    endpoints are intentionally unsupported; configure an LLM provider instead.
    """
    if not text or not text.strip():
        return ""

    raise TranslationUnavailableError(
        "No configured LLM translation provider is available. Configure OpenAI, Groq, or Gemini."
    )


# ==============================================================================
# LLM-POWERED TRANSLATION
# ==============================================================================

def _get_llm_client(
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """Safely get the LLM client, returning None if unavailable."""
    try:
        from coherex.llm import get_llm_client
        return get_llm_client(
            provider=provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            openai_model=model if provider == "openai" else None,
            groq_model=model if provider == "groq" else None,
            gemini_model=model if provider == "gemini" else None,
        )
    except ImportError:
        return None


def llm_translate_text(
    text: str,
    source_lang: str = "ar",
    target_lang: str = "en",
    context: str = "",
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Translate a single text using the LLM with dialect awareness.
    Returns None if the configured LLM is unavailable.
    """
    client = _get_llm_client(provider, gemini_api_key, openai_api_key, groq_api_key, model)
    if client is None or not client.is_available():
        return None

    try:
        return client.translate_text(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
        )
    except Exception as e:
        logger.warning("LLM single translation failed: %s", e)
        return None


def llm_translate_batch(
    texts: List[str],
    source_lang: str = "ar",
    target_lang: str = "en",
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Translate a batch of subtitle segments using the LLM with cross-segment coherence.
    Returns None if the configured LLM is unavailable.
    """
    client = _get_llm_client(provider, gemini_api_key, openai_api_key, groq_api_key, model)
    if client is None or not client.is_available():
        return None

    try:
        return client.translate_batch(
            texts=texts,
            source_lang=source_lang,
            target_lang=target_lang,
        )
    except Exception as e:
        logger.warning("LLM batch translation failed: %s", e)
        return None


# ==============================================================================
# PUBLIC TRANSLATION API
# ==============================================================================

def translate_segments(
    segments: List[Dict[str, Any]],
    source_lang: str = "auto",
    target_lang: str = "en",
    include_original: bool = False,
    use_llm: bool = True,
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Translates transcript segments with context awareness and timing preservation.
    
    Attempts LLM batch translation first for cross-segment coherence and dialect
    awareness. If it is unavailable, translation fails clearly unless an
    administrator explicitly enabled the legacy public endpoint.
    
    Args:
        segments: List of segment dicts with 'text', 'start', 'end' keys.
        source_lang: Source language code.
        target_lang: Target language code.
        include_original: If True, creates bilingual text (translated + original).
        use_llm: Whether to attempt LLM translation.
        gemini_api_key: Optional explicit API key.
        
    Returns:
        Deep copy of segments with source ``text`` preserved and a separate
        ``translated_text`` field.
    """
    if not segments:
        return []

    translated_segments = copy.deepcopy(segments)

    # Clean ASR artifacts and store originals
    for seg in translated_segments:
        raw_text = seg.get("text", "").strip()
        cleaned_text = clean_asr_repetitions(raw_text)
        seg["original_text"] = cleaned_text or raw_text

    # Try LLM batch translation first
    llm_success = False
    if use_llm:
        original_texts = [seg["original_text"] for seg in translated_segments]
        non_empty = [t for t in original_texts if t.strip()]

        if non_empty:
            llm_results = llm_translate_batch(
                texts=original_texts,
                source_lang=source_lang,
                target_lang=target_lang,
                provider=provider,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                groq_api_key=groq_api_key,
                model=model,
            )

            if llm_results:
                validate_translation_batch(original_texts, llm_results, target_lang)
                for seg, trans in zip(translated_segments, llm_results):
                    seg["translated_text"] = polish_translated_text(trans.strip())
                    seg["translation_engine"] = "configured-llm"
                llm_success = True
                logger.info(
                    "LLM batch translation completed: %d segments [%s -> %s]",
                    len(original_texts), source_lang, target_lang
                )

    # A network fallback is permitted only when explicitly enabled.
    if not llm_success:
        logger.warning("No valid LLM translation completed; refusing to label source text as translated")
        for seg in translated_segments:
            trans_text = raw_neural_translate(
                seg["original_text"],
                source_lang=source_lang,
                target_lang=target_lang
            )
            seg["translated_text"] = trans_text
            seg["translation_engine"] = "legacy-google-translate"

    # Keep the source transcript immutable. Renderers choose the appropriate track.
    for seg in translated_segments:
        seg["text"] = seg["original_text"]
        seg["display_text"] = (
            f"{seg['translated_text']}\n{seg['original_text']}"
            if include_original else seg["translated_text"]
        )

    return translated_segments


def translate_result(
    result: Dict[str, Any],
    target_lang: str = "en",
    source_lang: Optional[str] = None,
    bilingual: bool = False,
    use_llm: bool = True,
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Translates full CohereX result dictionary into target language.
    
    Args:
        result: Full transcription result dict.
        target_lang: Target language code.
        source_lang: Source language code (auto-detected if None).
        bilingual: If True, creates dual-track text.
        use_llm: Whether to attempt LLM translation.
        gemini_api_key: Optional explicit API key.
        
    Returns:
        Deep copy of result with translated segments.
    """
    new_result = copy.deepcopy(result)
    src = source_lang or result.get("language", "auto")

    if "segments" in new_result:
        new_result["segments"] = translate_segments(
            new_result["segments"],
            source_lang=src,
            target_lang=target_lang,
            include_original=bilingual,
            use_llm=use_llm,
            provider=provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            model=model,
        )

    new_result["target_language"] = target_lang
    new_result["is_bilingual"] = bilingual
    return new_result
