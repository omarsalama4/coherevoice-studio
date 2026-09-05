#!/usr/bin/env python3
"""
Google Gemini LLM client for CohereX.
Implements professional Meeting Minutes generation and dialect-aware translation
using Google Gemini 2.0 Flash via the google-genai SDK.
"""

import os
import json
import logging
from typing import List, Optional

from coherex.llm.client import LLMClient
from coherex.llm.schemas import (
    MeetingMinutes,
    ActionItem,
    Decision,
    TopicDiscussion,
)
from coherex.llm.prompts import (
    MOM_SYSTEM_PROMPT,
    MOM_USER_TEMPLATE,
    TRANSLATION_SYSTEM_PROMPT,
    SUBTITLE_TRANSLATION_TEMPLATE,
    MEETING_NOTES_TRANSLATION_TEMPLATE,
)

logger = logging.getLogger("coherex.llm.gemini")

# Language code to full name mapping for prompts
_LANG_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ko": "Korean",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "el": "Greek",
    "sv": "Swedish",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "cs": "Czech",
    "ro": "Romanian",
    "uk": "Ukrainian",
    "iw": "Hebrew",
    "fa": "Persian",
    "ur": "Urdu",
}


def _lang_name(code: str) -> str:
    """Convert a language code to a human-readable name."""
    return _LANG_NAMES.get(code, code)


class GeminiClient(LLMClient):
    """
    Google Gemini 2.0 Flash LLM client.
    
    Uses the free-tier google-genai SDK for:
    - Structured MOM generation with JSON schema enforcement
    - Batch subtitle translation with cross-segment coherence
    - Meeting notes translation preserving Markdown formatting
    
    Free tier limits: 15 RPM, 1M tokens/min, 1,500 requests/day.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model
        self._client = None

        if self._api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                logger.info("Gemini client initialized with model: %s", self._model)
            except Exception as e:
                logger.warning("Failed to initialize Gemini client: %s", e)
                self._client = None

    def is_available(self) -> bool:
        """Check if Gemini API key is configured and client is initialized."""
        return self._client is not None and bool(self._api_key)

    # ------------------------------------------------------------------
    # Meeting Minutes Generation
    # ------------------------------------------------------------------
    def generate_mom(
        self,
        transcript: str,
        meeting_date: str,
        duration: str = "",
        num_speakers: int = 0,
        source_language: str = "Arabic",
    ) -> MeetingMinutes:
        """
        Generate structured Meeting Minutes from a diarized transcript.
        Sends the full transcript in a single prompt (Gemini has a 1M token context window).
        """
        if not self.is_available():
            raise RuntimeError("Gemini client is not available. Check API key.")

        from google import genai
        from google.genai import types

        user_prompt = MOM_USER_TEMPLATE.format(
            meeting_date=meeting_date,
            duration=duration or "Unknown",
            num_speakers=num_speakers or "Unknown",
            source_language=source_language,
            transcript=transcript,
        )

        logger.info(
            "Generating MOM via Gemini [model=%s, transcript_chars=%d, speakers=%s]",
            self._model, len(transcript), num_speakers
        )

        # Build the response schema as a dict for Gemini's structured output
        mom_schema = {
            "type": "object",
            "properties": {
                "meeting_title": {"type": "string"},
                "meeting_date": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "executive_summary": {"type": "string"},
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "summary": {"type": "string"},
                            "speakers_involved": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["topic", "summary"],
                    },
                },
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string"},
                            "decided_by": {"type": "string"},
                            "timestamp": {"type": "string"},
                        },
                        "required": ["decision", "decided_by", "timestamp"],
                    },
                },
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "owner": {"type": "string"},
                            "deadline": {"type": "string"},
                            "priority": {"type": "string"},
                            "timestamp": {"type": "string"},
                        },
                        "required": ["task", "owner", "priority", "timestamp"],
                    },
                },
                "open_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "meeting_title",
                "meeting_date",
                "attendees",
                "executive_summary",
                "topics",
                "decisions",
                "action_items",
                "open_questions",
            ],
        }

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=MOM_SYSTEM_PROMPT + "\n\n" + user_prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=mom_schema,
                    temperature=0.15,
                    top_p=0.9,
                ),
            )

            raw_json = response.text
            logger.info("Gemini MOM response received (%d chars)", len(raw_json))

            data = json.loads(raw_json)
            return MeetingMinutes(**data)

        except Exception as e:
            logger.error("Gemini MOM generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Gemini MOM generation failed: {e}") from e

    # ------------------------------------------------------------------
    # Text Translation (single segment with context)
    # ------------------------------------------------------------------
    def translate_text(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        context: str = "",
    ) -> str:
        """Translate a single text segment with optional surrounding context."""
        if not self.is_available():
            raise RuntimeError("Gemini client is not available.")

        from google import genai
        from google.genai import types

        src_name = _lang_name(source_lang)
        tgt_name = _lang_name(target_lang)

        system = TRANSLATION_SYSTEM_PROMPT.format(target_language=tgt_name)

        user_msg = f"Translate from {src_name} to {tgt_name}:\n\n"
        if context:
            user_msg += f"[Context: {context}]\n\n"
        user_msg += f'"{text}"'

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=system + "\n\n" + user_msg)],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
            result = response.text.strip().strip('"').strip("'")
            return result
        except Exception as e:
            logger.warning("Gemini translate_text failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Batch Subtitle Translation
    # ------------------------------------------------------------------
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> List[str]:
        """
        Translate a batch of subtitle segments with cross-segment coherence.
        Sends segments in groups of 15 to balance context and API limits.
        """
        if not self.is_available():
            raise RuntimeError("Gemini client is not available.")
        if not texts:
            return []

        from google import genai
        from google.genai import types

        src_name = _lang_name(source_lang)
        tgt_name = _lang_name(target_lang)

        system = TRANSLATION_SYSTEM_PROMPT.format(target_language=tgt_name)

        batch_size = 15
        all_translations = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]

            segments_formatted = "\n".join(
                f"[{i+1}] {t}" for i, t in enumerate(batch)
            )

            user_prompt = SUBTITLE_TRANSLATION_TEMPLATE.format(
                source_language=src_name,
                target_language=tgt_name,
                segments_text=segments_formatted,
            )

            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=system + "\n\n" + user_prompt)],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )

                raw = response.text.strip()
                translations = json.loads(raw)

                if isinstance(translations, list):
                    # Ensure we have the right number of translations
                    while len(translations) < len(batch):
                        translations.append(batch[len(translations)])
                    all_translations.extend(translations[: len(batch)])
                else:
                    logger.warning("Unexpected Gemini batch response format, falling back")
                    all_translations.extend(batch)

            except Exception as e:
                logger.warning(
                    "Gemini batch translation failed for batch %d-%d: %s",
                    batch_start, batch_start + len(batch), e
                )
                # Fall back: return originals for this batch
                all_translations.extend(batch)

        return all_translations

    # ------------------------------------------------------------------
    # Meeting Notes Translation
    # ------------------------------------------------------------------
    def translate_meeting_notes(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> str:
        """Translate meeting notes markdown preserving formatting and professional tone."""
        if not self.is_available():
            raise RuntimeError("Gemini client is not available.")

        from google import genai
        from google.genai import types

        src_name = _lang_name(source_lang)
        tgt_name = _lang_name(target_lang)

        system = TRANSLATION_SYSTEM_PROMPT.format(target_language=tgt_name)
        user_prompt = MEETING_NOTES_TRANSLATION_TEMPLATE.format(
            source_language=src_name,
            target_language=tgt_name,
            text=text,
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=system + "\n\n" + user_prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.warning("Gemini meeting notes translation failed: %s", e)
            raise
