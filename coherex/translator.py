#!/usr/bin/env python3
"""
General-Purpose Dynamic Subtitle & Transcript Translator for CohereX.

Two translation engines:
1. LLM-Powered (Gemini): Context-aware, dialect-savvy batch translation that
   understands Egyptian Arabic idioms, code-switching, and conversational tone.
2. Fast Fallback (Google GTX): Per-segment neural translation via the Google
   Translate GTX endpoint with deep-translator backup.

The module auto-detects LLM availability and falls back gracefully.
"""

import re
import os
import sys
import copy
import json
import time
import logging
import urllib.parse
import urllib.request
from pathlib import Path
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


# ==============================================================================
# FAST FALLBACK: Google GTX Translation
# ==============================================================================

def raw_neural_translate(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    High-speed, general-purpose neural translation using Google Translate GTX engine
    with fallback to deep-translator. Completely dynamic and free of hardcoded phrase maps.
    """
    if not text or not text.strip():
        return ""

    sl = "auto" if source_lang in ["auto", None, ""] else source_lang
    tl = target_lang.split("-")[0] if "-" in target_lang and target_lang not in ["zh-CN", "zh-TW"] else target_lang

    try:
        url = (
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl="
            + urllib.parse.quote(sl)
            + "&tl="
            + urllib.parse.quote(tl)
            + "&dt=t&q="
            + urllib.parse.quote(text.strip())
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=12) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            if res_json and isinstance(res_json, list) and len(res_json) > 0 and res_json[0]:
                trans = "".join([part[0] for part in res_json[0] if part and part[0]])
                if trans.strip():
                    return polish_translated_text(trans.strip())
    except Exception:
        pass

    try:
        from deep_translator import GoogleTranslator
        src = "auto" if sl == "auto" else sl
        trans = GoogleTranslator(source=src, target=tl).translate(text.strip())
        if trans:
            return polish_translated_text(trans.strip())
    except Exception:
        pass

    return text.strip()


# ==============================================================================
# LLM-POWERED TRANSLATION
# ==============================================================================

def _get_llm_client(api_key: Optional[str] = None):
    """Safely get the LLM client, returning None if unavailable."""
    try:
        from coherex.llm import get_llm_client
        return get_llm_client(api_key=api_key)
    except ImportError:
        return None


def llm_translate_text(
    text: str,
    source_lang: str = "ar",
    target_lang: str = "en",
    context: str = "",
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Translate a single text using the LLM with dialect awareness.
    Returns None if LLM is unavailable (caller should fall back to GTX).
    """
    client = _get_llm_client(api_key)
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
    api_key: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Translate a batch of subtitle segments using the LLM with cross-segment coherence.
    Returns None if LLM is unavailable (caller should fall back to per-segment GTX).
    """
    client = _get_llm_client(api_key)
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
# PUBLIC TRANSLATION API (Auto-selects LLM or GTX)
# ==============================================================================

def translate_segments(
    segments: List[Dict[str, Any]],
    source_lang: str = "auto",
    target_lang: str = "en",
    include_original: bool = False,
    use_llm: bool = True,
    gemini_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Translates transcript segments with context awareness and timing preservation.
    
    Attempts LLM batch translation first for cross-segment coherence and dialect
    awareness, then falls back to per-segment Google GTX translation.
    
    Args:
        segments: List of segment dicts with 'text', 'start', 'end' keys.
        source_lang: Source language code.
        target_lang: Target language code.
        include_original: If True, creates bilingual text (translated + original).
        use_llm: Whether to attempt LLM translation.
        gemini_api_key: Optional explicit API key.
        
    Returns:
        Deep copy of segments with 'translated_text' and updated 'text' fields.
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
                api_key=gemini_api_key,
            )

            if llm_results and len(llm_results) == len(original_texts):
                for seg, trans in zip(translated_segments, llm_results):
                    seg["translated_text"] = polish_translated_text(trans.strip()) if trans else seg["original_text"]
                    seg["translation_engine"] = "gemini"
                llm_success = True
                logger.info(
                    "LLM batch translation completed: %d segments [%s -> %s]",
                    len(original_texts), source_lang, target_lang
                )

    # Fallback to per-segment GTX translation
    if not llm_success:
        logger.info("Using GTX per-segment translation for %d segments", len(translated_segments))
        for seg in translated_segments:
            trans_text = raw_neural_translate(
                seg["original_text"],
                source_lang=source_lang,
                target_lang=target_lang
            )
            seg["translated_text"] = trans_text
            seg["translation_engine"] = "gtx"

    # Set the display text
    for seg in translated_segments:
        if include_original:
            seg["text"] = f"{seg['translated_text']}\n{seg['original_text']}"
        else:
            seg["text"] = seg["translated_text"]

    return translated_segments


def translate_result(
    result: Dict[str, Any],
    target_lang: str = "en",
    source_lang: Optional[str] = None,
    bilingual: bool = False,
    use_llm: bool = True,
    gemini_api_key: Optional[str] = None,
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
            gemini_api_key=gemini_api_key,
        )

    new_result["target_language"] = target_lang
    new_result["is_bilingual"] = bilingual
    return new_result
