#!/usr/bin/env python3
"""
Abstract LLM client interface for CohereX.
Defines the contract that all LLM backends must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from coherex.llm.schemas import MeetingMinutes

logger = logging.getLogger("coherex.llm")


class LLMClient(ABC):
    """
    Abstract base class for LLM backends.
    
    Implementations provide:
    - Professional Meeting Minutes generation from diarized transcripts
    - Context-aware, dialect-savvy translation
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is configured and reachable."""
        ...

    @abstractmethod
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
        
        Args:
            transcript: Formatted diarized transcript with timestamps and speaker labels.
            meeting_date: Meeting date as YYYY-MM-DD for resolving relative dates.
            duration: Human-readable duration string (e.g., "26:52").
            num_speakers: Number of detected speakers.
            source_language: Primary language of the audio.
            
        Returns:
            MeetingMinutes: Structured meeting minutes with decisions, actions, topics.
        """
        ...

    @abstractmethod
    def translate_text(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        context: str = "",
    ) -> str:
        """
        Translate a single text with optional surrounding context.
        
        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            context: Optional surrounding text for coherence.
            
        Returns:
            Translated text string.
        """
        ...

    @abstractmethod
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> List[str]:
        """
        Translate a batch of texts (e.g., subtitle segments) with cross-segment coherence.
        
        Args:
            texts: List of text segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            
        Returns:
            List of translated strings in the same order.
        """
        ...

    @abstractmethod
    def translate_meeting_notes(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> str:
        """
        Translate meeting notes/MOM markdown preserving formatting and professional tone.
        
        Args:
            text: Full meeting notes markdown text.
            source_lang: Source language code.
            target_lang: Target language code.
            
        Returns:
            Translated markdown text.
        """
        ...
