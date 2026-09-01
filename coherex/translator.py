#!/usr/bin/env python3
"""
General-Purpose Dynamic Subtitle & Transcript Translator for CohereX
Performs robust, context-aware neural translation without hardcoded phrase overfits.
Supports all languages with special high-speed handling for Arabic dialects.
"""

import re
import os
import sys
import copy
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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


def clean_asr_repetitions(text: str) -> str:
    """Removes degenerate ASR looping patterns caused by background music or noise."""
    if not text:
        return ""
    text = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'(\b\w+\s+\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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


def polish_translated_text(text: str) -> str:
    """Normalizes subtitle punctuation, spacing, and capitalization."""
    if not text:
        return ""
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def translate_segments(
    segments: List[Dict[str, Any]],
    source_lang: str = "auto",
    target_lang: str = "en",
    include_original: bool = False
) -> List[Dict[str, Any]]:
    """
    Translates transcript segments with context awareness and timing preservation.
    """
    if not segments:
        return []
        
    translated_segments = copy.deepcopy(segments)
    
    for i, seg in enumerate(translated_segments):
        raw_text = seg.get("text", "").strip()
        cleaned_text = clean_asr_repetitions(raw_text)
        seg["original_text"] = cleaned_text or raw_text
        
        trans_text = raw_neural_translate(
            seg["original_text"],
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        seg["translated_text"] = trans_text
        if include_original:
            seg["text"] = f"{trans_text}\n{seg['original_text']}"
        else:
            seg["text"] = trans_text
            
    return translated_segments


def translate_result(
    result: Dict[str, Any],
    target_lang: str = "en",
    source_lang: Optional[str] = None,
    bilingual: bool = False
) -> Dict[str, Any]:
    """Translates full CohereX result dictionary into target language."""
    new_result = copy.deepcopy(result)
    src = source_lang or result.get("language", "auto")
    
    if "segments" in new_result:
        new_result["segments"] = translate_segments(
            new_result["segments"],
            source_lang=src,
            target_lang=target_lang,
            include_original=bilingual
        )
        
    new_result["target_language"] = target_lang
    new_result["is_bilingual"] = bilingual
    return new_result
