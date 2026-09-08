#!/usr/bin/env python3
"""
OpenAI and hosted OpenAI-compatible LLM client for CohereX.
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from coherex.llm.client import LLMClient
from coherex.llm.schemas import MeetingMinutes, ActionItem, Decision, TopicDiscussion

logger = logging.getLogger("coherex.llm.openai")

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient(LLMClient):
    """
    Client for OpenAI and hosted OpenAI-compatible endpoints.
    Uses standard HTTP/JSON requests with resilient error handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)).rstrip("/")
        self.provider_name = "openai" if self.base_url == DEFAULT_OPENAI_BASE_URL else "openai-compatible"

    def is_available(self) -> bool:
        """Check if API key is present."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Send a Responses API request to OpenAI, or Chat Completions to compatible endpoints."""
        is_official_openai = self.base_url.rstrip("/") == DEFAULT_OPENAI_BASE_URL
        url = f"{self.base_url}/responses" if is_official_openai else f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if is_official_openai:
            instructions = "\n\n".join(
                message["content"] for message in messages if message.get("role") == "system"
            )
            input_messages = [message for message in messages if message.get("role") != "system"]
            payload: Dict[str, Any] = {
                "model": self.model,
                "input": input_messages,
                "max_output_tokens": max_tokens,
                "store": False,
            }
            if instructions:
                payload["instructions"] = instructions
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res = json.loads(response.read().decode("utf-8"))
                if is_official_openai:
                    if isinstance(res.get("output_text"), str):
                        return res["output_text"].strip()
                    text_parts = []
                    for item in res.get("output", []):
                        for content in item.get("content", []):
                            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                                text_parts.append(content["text"])
                    result_text = "\n".join(text_parts).strip()
                    if not result_text:
                        raise RuntimeError("OpenAI Responses API returned no output text")
                    return result_text
                choices = res.get("choices", [])
                return choices[0].get("message", {}).get("content", "").strip() if choices else ""
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logger.error("OpenAI API HTTPError %d: %s", e.code, err_body)
            raise RuntimeError(f"OpenAI API error ({e.code}): {err_body}") from e
        except Exception as e:
            logger.error("OpenAI request failed: %s", e)
            raise RuntimeError(f"OpenAI connection error: {e}") from e

    def generate_mom(
        self,
        transcript: str,
        meeting_date: str = "",
        duration: str = "",
        num_speakers: int = 0,
        source_language: str = "Arabic (Egyptian Dialect)",
    ) -> MeetingMinutes:
        """Generate structured MOM using OpenAI."""
        markdown = self.generate_mom_markdown(
            transcript=transcript,
            meeting_date=meeting_date,
            duration=duration,
            num_speakers=num_speakers,
            source_language=source_language
        )
        title_match = re.search(r'# Minutes of Meeting:\s*(.+)', markdown)
        title = title_match.group(1).strip() if title_match else "Meeting Minutes"
        return MeetingMinutes(
            meeting_title=title,
            meeting_date=meeting_date or "2026-09-05",
            attendees=[],
            executive_summary=markdown[:600],
            topics=[],
            decisions=[],
            action_items=[],
            open_questions=[],
        )

    def generate_mom_markdown(
        self,
        transcript: str,
        meeting_date: str = "",
        duration: str = "",
        num_speakers: int = 0,
        source_language: str = "Arabic (Egyptian Dialect)",
    ) -> str:
        """Generate adaptive, professional MOM Markdown using OpenAI."""
        from coherex.llm.prompts import ADAPTIVE_MOM_SYSTEM_PROMPT, ADAPTIVE_MOM_USER_TEMPLATE

        system = ADAPTIVE_MOM_SYSTEM_PROMPT
        user = ADAPTIVE_MOM_USER_TEMPLATE.format(
            meeting_date=meeting_date,
            duration=duration,
            num_speakers=num_speakers,
            source_language=source_language,
            transcript=transcript
        )

        return self._chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.25,
            max_tokens=4096,
        )

    def translate_text(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        context: str = "",
    ) -> str:
        """Translate text with dialect awareness using OpenAI."""
        from coherex.llm.prompts import TRANSLATION_SYSTEM_PROMPT
        system = TRANSLATION_SYSTEM_PROMPT.format(target_language=target_lang)
        user = f"Context: {context}\n\nTranslate to {target_lang}: {text}" if context else f"Translate to {target_lang}: {text}"
        return self._chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.2,
            max_tokens=512,
        ).strip('"')

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> List[str]:
        """Batch subtitle translation with OpenAI."""
        from coherex.llm.prompts import SUBTITLE_TRANSLATION_TEMPLATE

        batch_size = 15
        all_translations = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            formatted_segments = "\n".join(f"[{idx+1}] {t}" for idx, t in enumerate(chunk))
            prompt = SUBTITLE_TRANSLATION_TEMPLATE.format(
                source_language=source_lang,
                target_language=target_lang,
                segments_text=formatted_segments,
            )

            try:
                response = self._chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a professional subtitle translator. Output a JSON object with key 'translations' containing an array of translated strings."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    json_mode=True,
                )
                cleaned_response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
                data = json.loads(cleaned_response)
                translations = data.get("translations", [])
                if isinstance(translations, list) and len(translations) == len(chunk):
                    all_translations.extend(translations)
                else:
                    raise RuntimeError("OpenAI returned the wrong number of subtitle translations")
            except Exception as e:
                logger.warning("OpenAI batch translation chunk failed: %s", e)
                raise RuntimeError("OpenAI batch translation failed") from e

        return all_translations

    def translate_meeting_notes(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> str:
        """Translate markdown meeting notes."""
        system = f"You are an executive translator. Translate the following markdown document from {source_lang} to {target_lang}. Preserve all headings, tables, and formatting."
        return self._chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text}
            ],
            temperature=0.2,
            max_tokens=4096,
        )
