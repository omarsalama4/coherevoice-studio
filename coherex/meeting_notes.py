#!/usr/bin/env python3
"""
Professional Meeting Notes & Transcript Summarizer for CohereVoice Studio.

Primary generation uses an API-backed LLM for structured MOM with executive
summary, decisions, action items, topics, and open questions. A legacy heuristic
renderer remains available only through explicit opt-in.
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from coherex.translator import clean_asr_repetitions

logger = logging.getLogger("coherex.meeting_notes")


# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

def format_short_time(seconds: float) -> str:
    """Format all evidence references as canonical HH:MM:SS."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def normalize_speaker_name(speaker_tag: Optional[str]) -> str:
    """Normalizes SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, etc."""
    if not speaker_tag:
        return "Speaker"
    match = re.match(r'SPEAKER_(\d+)', speaker_tag, re.IGNORECASE)
    if match:
        idx = int(match.group(1)) + 1
        return f"Speaker {idx}"
    return speaker_tag


def group_speaker_dialogue(
    segments: List[Dict[str, Any]],
    use_translated: bool = False,
    target_lang: str = "en"
) -> List[Dict[str, Any]]:
    """
    Combines consecutive segments from the same speaker into cohesive conversational paragraphs.
    Strictly separates translated text from original text without mixing languages.
    """
    if not segments:
        return []

    grouped = []
    current_turn = None

    for seg in segments:
        raw_spk = seg.get("speaker", "Speaker")
        spk = normalize_speaker_name(raw_spk)
        st = seg.get("start", 0.0)
        et = seg.get("end", st + 2.0)

        orig_text = clean_asr_repetitions(seg.get("original_text", seg.get("text", "")).strip())
        trans_text = clean_asr_repetitions(seg.get("translated_text", "").strip())

        if use_translated:
            if not trans_text and orig_text:
                raise RuntimeError(
                    "Translated meeting-note input is missing translated_text; "
                    "translate the result with a configured LLM first."
                )
            text = trans_text
        else:
            text = orig_text

        if not text:
            continue

        if current_turn and current_turn["speaker"] == spk and (st - current_turn["end"] < 3.5):
            current_turn["end"] = et
            current_turn["text"] += " " + text
        else:
            if current_turn:
                grouped.append(current_turn)
            current_turn = {
                "speaker": spk,
                "start": st,
                "end": et,
                "text": text
            }

    if current_turn:
        grouped.append(current_turn)

    return grouped


# ==============================================================================
# LLM-POWERED MOM GENERATION
# ==============================================================================

def _format_transcript_for_llm(segments: List[Dict[str, Any]]) -> str:
    """
    Formats diarized segments into a clean timestamped transcript for LLM consumption.
    
    Output format:
        [00:02:15] Speaker 1: Let's review the sprint deliverables.
        [00:02:24] Speaker 2: اه شغالين تمام، بس في bug في الـ edge case.
    """
    grouped = group_speaker_dialogue(segments, use_translated=False)
    lines = []
    for turn in grouped:
        ts = format_short_time(turn["start"])
        speaker = turn["speaker"]
        text = turn["text"].strip()
        if text:
            lines.append(f"[{ts}] {speaker}: {text}")
    return "\n".join(lines)


def _format_mom_as_markdown(mom) -> str:
    """
    Converts a MeetingMinutes Pydantic object into polished, professional Markdown.
    """
    md = []

    # Header
    md.append(f"# 📋 {mom.meeting_title}")
    attendees_str = ", ".join(mom.attendees) if mom.attendees else "Unknown"
    md.append(f"**Date:** {mom.meeting_date} | **Attendees ({len(mom.attendees)}):** {attendees_str}")
    md.append("---")

    # Executive Summary
    md.append("## 📌 Executive Summary")
    md.append(mom.executive_summary)
    md.append("")

    # Key Topics & Discussions
    if mom.topics:
        md.append("## 💡 Key Discussion Topics")
        for i, topic in enumerate(mom.topics, 1):
            md.append(f"### {i}. {topic.topic}")
            md.append(topic.summary)
            if topic.speakers_involved:
                speakers = ", ".join(topic.speakers_involved)
                md.append(f"*Speakers: {speakers}*")
            md.append("")

    # Decisions Made
    if mom.decisions:
        md.append("## ✅ Decisions Made")
        md.append("| # | Decision | Decided By | Timestamp |")
        md.append("|---|----------|------------|-----------|")
        for i, d in enumerate(mom.decisions, 1):
            md.append(f"| {i} | {d.decision} | {d.decided_by} | `{d.timestamp}` |")
        md.append("")

    # Action Items
    if mom.action_items:
        md.append("## 🎯 Action Items")
        md.append("| # | Task | Owner | Deadline | Priority | Ref |")
        md.append("|---|------|-------|----------|----------|-----|")
        for i, a in enumerate(mom.action_items, 1):
            deadline = a.deadline or "TBD"
            priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(a.priority, "⚪")
            md.append(f"| {i} | {a.task} | {a.owner} | {deadline} | {priority_emoji} {a.priority} | `{a.timestamp}` |")
        md.append("")

    # Open Questions
    if mom.open_questions:
        md.append("## ❓ Open Questions & Parking Lot")
        for q in mom.open_questions:
            md.append(f"- {q}")
        md.append("")

    md.append("---")
    md.append("*Generated by CohereVoice Studio — AI-Powered Meeting Intelligence*")

    return "\n".join(md)


def _evidence_timestamps(segments: List[Dict[str, Any]]) -> set[str]:
    """Return the exact turn-start timestamps the model may cite as evidence."""
    return {
        format_short_time(turn["start"])
        for turn in group_speaker_dialogue(segments, use_translated=False)
    }


def _validate_structured_evidence(mom, allowed: set[str]) -> None:
    for kind, entries in (("decision", mom.decisions), ("action item", mom.action_items)):
        for entry in entries:
            timestamp = str(entry.timestamp).strip().strip("[]`")
            if timestamp not in allowed:
                raise ValueError(f"Ungrounded {kind} timestamp returned by LLM: {timestamp}")


def _validate_markdown_evidence(markdown: str, allowed: set[str]) -> None:
    required = ("Executive Summary", "Decisions", "Action Items", "Open Questions")
    missing = [heading for heading in required if heading.lower() not in markdown.lower()]
    if missing:
        raise ValueError("LLM meeting notes omitted required sections: " + ", ".join(missing))

    headings = re.findall(r"^##\s+(.+)$", markdown, flags=re.MULTILINE)
    core_terms = ("meeting overview", "executive summary", "decisions", "action items", "open questions")
    adaptive_headings = [
        heading for heading in headings
        if not any(term in heading.lower() for term in core_terms)
    ]
    if len(adaptive_headings) < 2:
        raise ValueError("LLM meeting notes omitted substantive adaptive deep-dive sections")
    for timestamp in re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", markdown):
        if timestamp not in allowed:
            raise ValueError(f"LLM returned an ungrounded evidence timestamp: {timestamp}")

    # Decisions and actions are the highest-risk sections for fabricated facts.
    # Require evidence in each unless the model explicitly reports that none
    # were identified. Merely validating timestamps that happen to be present
    # allowed an uncited hallucinated section to pass.
    for heading, next_heading in (
        ("Decisions", "Action Items"),
        ("Action Items", "Open Questions"),
    ):
        section_match = re.search(
            rf"^##[^\n]*{re.escape(heading)}[^\n]*\n(.*?)(?=^##\s|\Z)",
            markdown,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if not section_match:
            raise ValueError(f"LLM meeting notes omitted the {heading} section")
        section = section_match.group(1).strip()
        says_none = bool(
            re.search(
                r"\b(?:none|not identified|n/?a|no\s+(?:(?:final|formal|confirmed|explicit)\s+)?"
                r"(?:decisions?|actions?|action items?|commitments?))\b",
                section,
                re.I,
            )
        )
        citations = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", section)
        if section and not says_none and not citations:
            raise ValueError(f"LLM {heading} section contains claims without evidence timestamps")


def generate_llm_meeting_notes(
    result: Dict[str, Any],
    title: str = "Meeting Notes",
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    model: Optional[str] = None,
    meeting_date: Optional[str] = None,
    fail_closed: bool = False,
) -> Optional[str]:
    """
    Generate professional Meeting Minutes using the integrated LLM.
    
    Args:
        result: Full transcription result dict with segments.
        title: Optional title override.
        gemini_api_key: Optional explicit API key.
        meeting_date: Meeting date as YYYY-MM-DD.
        
    Returns:
        Professional Markdown MOM string, or None if LLM is unavailable.
    """
    try:
        from coherex.llm import get_llm_client
        client = get_llm_client(
            provider=provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            openai_model=model if provider == "openai" else None,
            groq_model=model if provider == "groq" else None,
            gemini_model=model if provider == "gemini" else None,
        )
        if client is None or not client.is_available():
            if fail_closed:
                raise RuntimeError("Required LLM API provider is unavailable")
            logger.info("LLM not available — returning None for heuristic fallback")
            return None
    except ImportError:
        logger.info("LLM package not available — returning None")
        return None

    segments = result.get("segments", [])
    if not segments:
        return None

    # Format transcript for LLM
    transcript_text = _format_transcript_for_llm(segments)
    allowed_timestamps = _evidence_timestamps(segments)

    # Determine meeting metadata
    if not meeting_date:
        meeting_date = datetime.now().strftime("%Y-%m-%d")

    total_duration = segments[-1].get("end", 0.0) if segments else 0.0
    duration_str = format_short_time(total_duration)

    raw_speakers = sorted(list(set(
        seg.get("speaker") for seg in segments if seg.get("speaker")
    )))
    num_speakers = len(raw_speakers)
    source_lang = result.get("language", "ar")
    lang_name = {"ar": "Arabic (Egyptian Dialect)", "en": "English"}.get(source_lang, source_lang)

    try:
        logger.info("Generating AI-powered MOM [speakers=%d, duration=%s]", num_speakers, duration_str)
        if hasattr(client, "generate_mom_markdown"):
            markdown = client.generate_mom_markdown(
                transcript=transcript_text,
                meeting_date=meeting_date,
                duration=duration_str,
                num_speakers=num_speakers,
                source_language=lang_name,
            )
            _validate_markdown_evidence(markdown, allowed_timestamps)
        else:
            mom = client.generate_mom(
                transcript=transcript_text,
                meeting_date=meeting_date,
                duration=duration_str,
                num_speakers=num_speakers,
                source_language=lang_name,
            )
            _validate_structured_evidence(mom, allowed_timestamps)
            markdown = _format_mom_as_markdown(mom)
        logger.info("AI MOM generated successfully (%d chars)", len(markdown))
        return markdown

    except Exception as e:
        if fail_closed:
            raise RuntimeError(f"Required LLM MOM generation failed validation: {e}") from e
        logger.error("LLM MOM generation failed: %s — falling back to heuristic", e)
        return None


# ==============================================================================
# HEURISTIC FALLBACK (Original behavior)
# ==============================================================================

def extract_key_points(grouped_turns: List[Dict[str, Any]], max_points: int = 8) -> List[str]:
    """Extracts key highlights and conversational points from dialogue turns."""
    points = []
    for turn in grouped_turns:
        text = turn["text"].strip()
        if len(text) > 40:
            sentences = re.split(r'[.!?؟،,\n]+', text)
            clean_s = [s.strip() for s in sentences if len(s.strip()) > 20]
            if clean_s:
                t_str = format_short_time(turn["start"])
                points.append(f"**[{t_str}] {turn['speaker']}**: {clean_s[0]}")
                if len(points) >= max_points:
                    break
    return points


def generate_heuristic_meeting_notes(
    result: Dict[str, Any],
    title: str = "Meeting & Conversation Notes",
    use_translated: bool = False,
    target_lang: str = "en"
) -> str:
    """
    Generates a rule-based Markdown report for Meeting Notes mode.
    Used as fallback when no LLM is available.
    """
    segments = result.get("segments", [])
    grouped_turns = group_speaker_dialogue(segments, use_translated=use_translated, target_lang=target_lang)

    total_duration = segments[-1].get("end", 0.0) if segments else 0.0
    duration_str = format_short_time(total_duration)

    raw_speakers = sorted(list(set(seg.get("speaker") for seg in segments if seg.get("speaker"))))
    speakers = [normalize_speaker_name(s) for s in raw_speakers]
    speaker_list_str = ", ".join(speakers) if speakers else "Single Speaker / Unlabeled"

    lang_code = (target_lang if use_translated else result.get("language", "ar")).upper()

    highlights = extract_key_points(grouped_turns)

    md = []
    md.append(f"# 📋 {title}")
    md.append(f"**Audio Duration:** {duration_str} | **Language:** {lang_code} | **Participants ({len(speakers)}):** {speaker_list_str}")
    md.append("---")

    # Executive Summary Section
    md.append("## 📌 Executive Summary")
    if grouped_turns:
        intro_snippet = " ".join(t["text"] for t in grouped_turns[:3])[:350]
        md.append(f"> {intro_snippet}...")
    else:
        md.append("_No dialogue recorded._")
    md.append("")

    # Key Highlights Section
    if highlights:
        md.append("## 💡 Key Discussion Highlights & Milestones")
        for pt in highlights:
            md.append(f"- {pt}")
        md.append("")

    # Topic Timeline & Index
    md.append("## ⏱️ Topic Timeline & Discussion Flow")
    for t in grouped_turns[:15]:
        t_start = format_short_time(t["start"])
        snippet = t["text"][:100] + ("..." if len(t["text"]) > 100 else "")
        md.append(f"- **`{t_start}`** — **{t['speaker']}**: {snippet}")
    if len(grouped_turns) > 15:
        md.append(f"- _...and {len(grouped_turns) - 15} additional conversation segments below._")
    md.append("")

    # Full Structured Dialogue Transcript
    md.append("## 💬 Full Speaker-by-Speaker Transcript")
    for t in grouped_turns:
        t_start = format_short_time(t["start"])
        t_end = format_short_time(t["end"])
        md.append(f"### 👤 {t['speaker']} `[{t_start} ➔ {t_end}]`")
        md.append(f"{t['text']}\n")

    return "\n".join(md)


# ==============================================================================
# PUBLIC API — Auto-selects LLM or Heuristic
# ==============================================================================

def generate_meeting_notes_markdown(
    result: Dict[str, Any],
    title: str = "Meeting & Conversation Notes",
    use_translated: bool = False,
    target_lang: str = "en",
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    model: Optional[str] = None,
    meeting_date: Optional[str] = None,
    use_llm: bool = True,
    allow_heuristic_fallback: bool = False,
) -> str:
    """
    Generates professional Meeting Notes from transcription results.
    
    Uses an API-backed LLM and fails closed by default if no provider is configured.
    
    Args:
        result: Full transcription result dict with segments.
        title: Report title.
        use_translated: Whether to use translated text in heuristic mode.
        target_lang: Target language for translation.
        gemini_api_key: Optional explicit Gemini API key.
        meeting_date: Meeting date as YYYY-MM-DD.
        use_llm: Whether to attempt LLM generation.
        allow_heuristic_fallback: Explicit opt-in for non-AI legacy output.
        
    Returns:
        Professional Markdown meeting notes string.
    """
    # Try LLM-powered generation first
    if use_llm:
        llm_result = generate_llm_meeting_notes(
            result=result,
            title=title,
            provider=provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            model=model,
            meeting_date=meeting_date,
            fail_closed=not allow_heuristic_fallback,
        )
        if llm_result is not None:
            return llm_result

    if not allow_heuristic_fallback:
        raise RuntimeError("MOM generation failed and heuristic fallback is disabled")

    # Fallback to heuristic generation
    logger.info("Using heuristic meeting notes generation")
    return generate_heuristic_meeting_notes(
        result=result,
        title=title,
        use_translated=use_translated,
        target_lang=target_lang,
    )
