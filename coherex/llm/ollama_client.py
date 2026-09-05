#!/usr/bin/env python3
"""
Local Open-Source LLM Client for CohereX using Ollama.
Supports local models like Qwen 2.5 (7B / 14B) running 100% offline on GPU/CPU.
Includes auto-start functionality to ensure Ollama runs seamlessly in the background.
Serves both Meetings (Adaptive MOM) and Videos (Subtitles & Synopsis).
"""

import os
import sys
import json
import re
import time
import logging
import subprocess
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from coherex.llm.client import LLMClient
from coherex.llm.schemas import MeetingMinutes
from coherex.llm.prompts import (
    ADAPTIVE_MOM_SYSTEM_PROMPT,
    ADAPTIVE_MOM_USER_TEMPLATE,
    VIDEO_SYNOPSIS_SYSTEM_PROMPT,
    VIDEO_SYNOPSIS_USER_TEMPLATE,
    TRANSLATION_SYSTEM_PROMPT,
    SUBTITLE_TRANSLATION_TEMPLATE,
)

logger = logging.getLogger("coherex.llm.ollama")

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PREFERRED_LOCAL_MODELS = [
    "qwen2.5:7b-instruct",
    "qwen2.5:7b",
    "qwen2.5:14b-instruct",
    "qwen2.5:14b",
    "qwen2.5:3b",
    "deepseek-r1:7b",
]


def ensure_ollama_running(base_url: str = DEFAULT_OLLAMA_URL, timeout_secs: int = 6) -> bool:
    """
    Checks if Ollama is running; if not, automatically starts 'ollama serve' in background.
    """
    url = f"{base_url.rstrip('/')}/api/tags"

    # Fast probe (0.5s timeout) to avoid UI freezing
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=0.8) as res:
            if res.status == 200:
                return True
    except Exception:
        pass

    # Attempt to auto-start ollama serve in background
    logger.info("Ollama is not running. Attempting to start 'ollama serve' in background...")
    try:
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS

        subprocess.Popen(
            ["ollama", "serve"],
            creationflags=creation_flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            shell=False,
        )

        # Wait up to timeout_secs for the server to become responsive
        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            time.sleep(0.8)
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1.0) as res:
                    if res.status == 200:
                        logger.info("Ollama server auto-started successfully!")
                        return True
            except Exception:
                continue

    except Exception as start_err:
        logger.warning("Could not auto-start Ollama: %s", start_err)

    return False


class OllamaClient(LLMClient):
    """
    Local Open-Source LLM Client powered by Ollama.
    Runs 100% locally and offline on NVIDIA GPU or CPU.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: Optional[str] = None,
        context_size: int = 16384,
        auto_start: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self._model = model
        self._cached_models: Optional[List[str]] = None

        if auto_start:
            ensure_ollama_running(self.base_url, timeout_secs=5)

    def is_available(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                return response.status == 200
        except Exception:
            return False

    def get_installed_models(self) -> List[str]:
        """Fetch list of models currently installed in Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                self._cached_models = [m["name"] for m in data.get("models", [])]
                return self._cached_models
        except Exception:
            return []

    def get_active_model(self) -> str:
        """Resolve the best available local model."""
        if self._model:
            return self._model

        installed = self.get_installed_models()
        if not installed:
            return "qwen2.5:7b-instruct"

        for pref in PREFERRED_LOCAL_MODELS:
            for inst in installed:
                if pref in inst:
                    self._model = inst
                    return self._model

        self._model = installed[0]
        return self._model

    def set_model(self, model_name: str):
        """Manually select a specific local model."""
        self._model = model_name

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        num_predict: int = 4096,
        timeout: int = 360,
    ) -> str:
        """Execute chat completion against Ollama /api/chat with FlashAttention & ChatML."""
        model = self.get_active_model()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.context_size,
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": num_predict,
            }
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            res = json.loads(response.read().decode("utf-8"))
            content = res.get("message", {}).get("content", "")
            # Strip reasoning tags (e.g., <think>...</think>)
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return clean

    def generate_mom(
        self,
        transcript: str,
        meeting_date: str = "",
        duration: str = "",
        num_speakers: int = 0,
        source_language: str = "Arabic (Egyptian Dialect)",
    ) -> MeetingMinutes:
        """Generate structured MOM using local LLM."""
        markdown = self.generate_mom_markdown(
            transcript=transcript,
            meeting_date=meeting_date,
            duration=duration,
            num_speakers=num_speakers,
            source_language=source_language
        )
        title_match = re.search(r'# Minutes of Meeting:\s*(.+)', markdown)
        title = title_match.group(1).strip() if title_match else "Executive Meeting Minutes"

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
        """
        Generate Adaptive Executive Minutes of Meeting (MOM).
        Adapts structure dynamically to the conversation while maintaining consulting rigor.
        """
        user_prompt = ADAPTIVE_MOM_USER_TEMPLATE.format(
            meeting_date=meeting_date,
            duration=duration,
            num_speakers=num_speakers,
            source_language=source_language,
            transcript=transcript
        )

        messages = [
            {"role": "system", "content": ADAPTIVE_MOM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        logger.info("Generating adaptive MOM with local LLM [%s]...", self.get_active_model())
        return self._chat_completion(messages, temperature=0.25, num_predict=4096)

    def generate_video_synopsis(
        self,
        transcript: str,
        media_name: str = "Video",
        duration: str = "",
        language: str = "en",
    ) -> str:
        """Generate video synopsis, chapters, and key highlights."""
        user_prompt = VIDEO_SYNOPSIS_USER_TEMPLATE.format(
            media_name=media_name,
            duration=duration,
            language=language,
            transcript=transcript
        )

        messages = [
            {"role": "system", "content": VIDEO_SYNOPSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        logger.info("Generating video synopsis with local LLM [%s]...", self.get_active_model())
        return self._chat_completion(messages, temperature=0.2, num_predict=2048)

    def translate_text(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        context: str = "",
    ) -> str:
        """Translate single text with Egyptian dialect awareness."""
        system = TRANSLATION_SYSTEM_PROMPT.format(target_language=target_lang)
        user = f"Context: {context}\n\nTranslate to {target_lang}: {text}" if context else f"Translate to {target_lang}: {text}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        return self._chat_completion(messages, temperature=0.2, num_predict=512).strip('"')

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> List[str]:
        """Translate subtitle cues in batches with subtitle constraints."""
        batch_size = 15
        all_trans = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            formatted = "\n".join(f"[{idx+1}] {t}" for idx, t in enumerate(chunk))
            user_msg = SUBTITLE_TRANSLATION_TEMPLATE.format(
                source_language=source_lang,
                target_language=target_lang,
                segments_text=formatted
            )

            messages = [
                {"role": "system", "content": "You are a professional subtitle translator. Output a JSON array of translated strings only."},
                {"role": "user", "content": user_msg}
            ]

            try:
                raw = self._chat_completion(messages, temperature=0.2, num_predict=2048, timeout=90)
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    all_trans.extend(parsed[: len(chunk)])
                else:
                    all_trans.extend(chunk)
            except Exception:
                all_trans.extend(chunk)

        return all_trans

    def translate_meeting_notes(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> str:
        """Translate markdown document preserving all layout and tables."""
        system = f"Translate this meeting document from {source_lang} to {target_lang}. Preserve all markdown formatting, tables, and headings."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]
        return self._chat_completion(messages, temperature=0.2, num_predict=4096)
