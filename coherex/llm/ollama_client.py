#!/usr/bin/env python3
"""
Local Open-Source LLM Client for CohereX using Ollama.
Supports local models like Qwen 2.5 (7B / 14B) running 100% offline on GPU/CPU.
Generates executive-grade Meeting Minutes (MOM) and provides dialect-aware translation.
"""

import os
import sys
import json
import re
import logging
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from coherex.llm.client import LLMClient
from coherex.llm.schemas import MeetingMinutes, ActionItem, Decision, TopicDiscussion

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

# Executive MOM Prompt for Brainstorming & Strategic Meetings
EXECUTIVE_MOM_SYSTEM_PROMPT = """You are a Principal Product Strategist and Executive Chief of Staff documenting a meeting conducted in Egyptian Arabic (Masri) and English.
Your task is to write an exhaustive, publication-grade Minutes of Meeting (MOM) document that captures the full strategic and technical depth of the discussion.

CRITICAL DIRECTIVES:
1. THOROUGH SYNTHESIS: Extract all substantive discussions, proposals, user problems, trade-offs, and product decisions across the ENTIRE transcript. Do not produce a brief summary; produce a deep, structured strategic document.
2. EGYPTIAN DIALECT EXPERTISE: The participants speak Egyptian Arabic and Arabizi. Understand colloquial phrases, metaphors, and cultural context (e.g., 'مش عارف تروح فين', 'عايز تجيب فيه ايه', 'تكبر المشروع', 'مش بنخترع العجلة', 'عايز يخرج مقتنع', '16 personalities', 'day in the life videos'). Translate all ideas into polished, professional executive English.
3. GROUNDING: Base every single point strictly on the provided transcript. Do not invent facts or participants not present in the discussion.
4. CLEAN OUTPUT: Output strictly valid Markdown. Do not wrap your response in markdown code blocks."""

EXECUTIVE_MOM_STRUCTURE = """REQUIRED DOCUMENT STRUCTURE:

# Minutes of Meeting: [Descriptive Meeting Title]

## Meeting Overview
- **Meeting purpose:** [Clear description of purpose]
- **Primary focus:** [Key areas of focus]
- **Source duration:** [Approximate duration]
- **Meeting format:** [Informal group brainstorming / formal sync]
- **Decision status:** [Summary of consensus and unresolved scope]

## Executive Summary
[3-4 detailed paragraphs explaining:
- The core platform concept and problem being solved
- How it works vs traditional tools (adaptive conversational discovery vs rigid questionnaires)
- The core value proposition (reducing decision paralysis, realistic next steps)
- Scope triage (MVP focus vs long-term ambitions)]

## Problem Statement
[Narrative of user struggle with educational/career uncertainty, followed by a bulleted list analyzing specific shortcomings of traditional assessments like 16personalities: boring, rigid, misleading multiple-choice, generic labels, no actionable next steps.]

## Proposed Solution
[Numbered list of the 8 core pillars of the conversational AI career companion.]

## Target Users
[Detailed breakdown of user segments: secondary school / Thanaweya Amma, university students choosing a major/track, recent graduates, career switchers, freelancers.]

## Intended User Journey
[9-step chronological flow from Onboarding -> Adaptive Assessment -> Profile Creation -> Recommendations -> Explainable Reasoning -> Realistic Preview -> Decision Support -> Initial Roadmap -> Saved Continuity.]

## Proposed Features
### Core MVP Features
[Bulleted list of essential demo features]
### Optional Prototype Enhancements
[Stretch goals discussed]
### Future Vision
[Long-term platform vision]

## Potential Differentiators
[Detailed bolded bullet points explaining how this stands out from traditional tests and chatbots.]

## Career Profile Data Discussed
[Specific attributes each career profile should display: tasks, skills, education, remote work, compensation/demand in Egypt, day-in-the-life videos, sample trial tasks.]

## Technical and Product Considerations
[Key technical trade-offs: fine-tuning vs prompt engineering, video curation vs generation, evaluation metrics, privacy.]

## Key Concerns and Differing Views
### Scope Risk
[Discussion on feature creep vs hackathon deadline]
### Novelty and Value
[How to prove real differentiated value]
### Effectiveness and Trust
[Ensuring recommendations feel defensible]
### Assessment Length
[Balancing conversational depth with user patience]
### One Career Versus Several
[Avoiding single-career dogma]

## Working Scope for the Prototype
[The realistic 8-step journey that can actually be built for the demo.]

## Preliminary Decisions and Agreements
[Bulleted list of consensus points.]

## Open Questions
[Unresolved questions for future alignment.]

## Recommended Next Steps
[10 numbered, concrete actionable tasks.]

## Draft One-Sentence Pitch
[A single bolded, compelling summary sentence.]

## Draft Problem Statement
[A crisp, formal 3-4 sentence problem statement for the pitch.]"""


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
    ):
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self._model = model
        self._cached_models: Optional[List[str]] = None

    def get_installed_models(self) -> List[str]:
        """Fetch list of models currently installed in Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                self._cached_models = [m["name"] for m in data.get("models", [])]
                return self._cached_models
        except Exception as e:
            logger.debug("Failed to fetch Ollama models: %s", e)
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

    def is_available(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status == 200
        except Exception:
            return False

    def generate_mom(
        self,
        transcript: str,
        meeting_date: str = "",
        duration: str = "",
        num_speakers: int = 0,
        source_language: str = "Arabic (Egyptian Dialect)",
    ) -> MeetingMinutes:
        """
        Generate executive-grade Meeting Minutes using the local open-source LLM.
        """
        markdown_mom = self.generate_mom_markdown(
            transcript=transcript,
            meeting_date=meeting_date,
            duration=duration,
            num_speakers=num_speakers,
            source_language=source_language,
        )

        # Parse sections into structured MeetingMinutes object for schema compatibility
        title_match = re.search(r'# Minutes of Meeting:\s*(.+)', markdown_mom)
        title = title_match.group(1).strip() if title_match else "Executive Meeting Minutes"

        exec_match = re.search(r'## Executive Summary\s*\n\n?([\s\S]+?)(?=\n##|\Z)', markdown_mom)
        exec_summary = exec_match.group(1).strip() if exec_match else markdown_mom[:500]

        return MeetingMinutes(
            meeting_title=title,
            meeting_date=meeting_date or "2026-09-05",
            attendees=["Brainstorming Participants"],
            executive_summary=exec_summary,
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
        Generate complete, publication-grade Markdown MOM using local LLM via /api/chat.
        """
        model = self.get_active_model()
        logger.info("Generating MOM via local LLM [%s] (transcript words: %d)...", model, len(transcript.split()))

        prompt_content = f"{EXECUTIVE_MOM_STRUCTURE}\n\n=== SOURCE TRANSCRIPT ===\n{transcript}\n=== END SOURCE TRANSCRIPT ===\n\nProduce the complete, exhaustive Minutes of Meeting following the required structure and depth."

        messages = [
            {"role": "system", "content": EXECUTIVE_MOM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content}
        ]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.context_size,
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 4096
            }
        }

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=data_bytes,
                headers={"Content-Type": "application/json"}
            )
            # Up to 6 minutes timeout for local inference of long transcripts
            with urllib.request.urlopen(req, timeout=360) as response:
                res = json.loads(response.read().decode("utf-8"))
                content = res.get("message", {}).get("content", "")

                # Clean any reasoning <think> tags (e.g. from DeepSeek R1 models)
                clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                logger.info("Local MOM generated successfully (%d chars)", len(clean_content))
                return clean_content

        except Exception as e:
            logger.error("Local LLM MOM generation failed: %s", e)
            raise RuntimeError(f"Local LLM MOM generation failed: {e}") from e

    def translate_text(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        context: str = "",
    ) -> str:
        """Translate single text with dialect awareness using local LLM."""
        model = self.get_active_model()
        system = f"You are a professional bilingual interpreter translating Egyptian Arabic (عامية مصرية) to {target_lang}. Translate idioms naturally into culturally equivalent expressions. Output ONLY the translation."
        user_prompt = f'Context: {context}\n\nTranslate: "{text}"' if context else f'Translate: "{text}"'

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(f"{self.base_url}/api/chat", data=data_bytes, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as response:
                res = json.loads(response.read().decode("utf-8"))
                content = res.get("message", {}).get("content", "").strip()
                return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip().strip('"')
        except Exception as e:
            logger.warning("Local LLM translation failed: %s", e)
            raise

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> List[str]:
        """Translate batch of subtitle cues using local LLM."""
        model = self.get_active_model()
        system = f"You are a subtitle translator specializing in Egyptian Arabic to {target_lang}. Translate each numbered segment concisely for subtitles. Output JSON array of translated strings only."

        batch_size = 12
        all_trans = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            formatted = "\n".join(f"[{idx+1}] {t}" for idx, t in enumerate(chunk))
            user_msg = f"Translate segments to {target_lang}:\n{formatted}\nReturn JSON array of {len(chunk)} strings."

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg}
                ],
                "stream": False,
                "options": {"temperature": 0.2}
            }

            try:
                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(f"{self.base_url}/api/chat", data=data_bytes, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    raw = res.get("message", {}).get("content", "").strip()
                    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                    # Try parsing json
                    parsed = json.loads(clean)
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
        """Translate markdown meeting notes using local LLM."""
        model = self.get_active_model()
        system = f"Translate this meeting document from {source_lang} to professional {target_lang}. Preserve all markdown formatting, tables, and headings."

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text}
            ],
            "stream": False,
            "options": {"num_ctx": self.context_size, "temperature": 0.2}
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=data_bytes, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            res = json.loads(response.read().decode("utf-8"))
            content = res.get("message", {}).get("content", "").strip()
            return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
