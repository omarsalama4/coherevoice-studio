#!/usr/bin/env python3
"""Deterministic subtitle cue generation and quality validation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def format_timestamp(seconds: float, is_vtt: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1_000)
    sep = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _words(text: str, line_limit: int) -> List[str]:
    result: List[str] = []
    for token in re.sub(r"\s+", " ", text.strip()).split(" "):
        result.extend(token[i:i + line_limit] for i in range(0, len(token), line_limit))
    return [word for word in result if word]


def _chunk_text(text: str, line_limit: int, max_lines: int) -> List[str]:
    """Pack text into cues whose wrapped lines satisfy both configured limits."""
    if line_limit < 1 or max_lines < 1:
        raise ValueError("Subtitle line limits must be positive")
    words = _words(text, line_limit)
    if not words:
        return []
    cues: List[str] = []
    cue_lines: List[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) <= line_limit:
            line = candidate
            continue
        if line:
            cue_lines.append(line)
        line = word
        if len(cue_lines) == max_lines:
            cues.append("\n".join(cue_lines))
            cue_lines = []
    if line:
        cue_lines.append(line)
    if cue_lines:
        cues.append("\n".join(cue_lines))
    return cues


def _partition_text(text: str, count: int, line_limit: int) -> List[str]:
    """Partition text into exactly ``count`` ordered, single-line groups."""
    chunks = _chunk_text(text, line_limit, 1)
    if len(chunks) > count:
        raise ValueError("Shared bilingual timing skeleton is too small for the text")
    return chunks + [""] * (count - len(chunks))


def _timed_cues(chunks: List[str], start: float, end: float, speaker: Optional[str]) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    start_ms = max(0, int(round(float(start) * 1000)))
    requested_end_ms = int(round(float(end) * 1000))
    end_ms = max(start_ms + len(chunks), requested_end_ms)
    weights = [max(1, len(chunk.replace("\n", " "))) for chunk in chunks]
    total = sum(weights)
    distributable_ms = end_ms - start_ms - len(chunks)
    elapsed_weight = 0
    cues: List[Dict[str, Any]] = []
    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        cue_start_ms = start_ms + index + round(distributable_ms * elapsed_weight / total)
        elapsed_weight += weight
        cue_end_ms = end_ms if index == len(chunks) - 1 else start_ms + index + 1 + round(distributable_ms * elapsed_weight / total)
        cues.append({"start": cue_start_ms / 1000, "end": cue_end_ms / 1000, "text": chunk, "speaker": speaker})
    return cues


def split_text_into_cues(
    text: str, start_time: float, end_time: float,
    max_chars_per_line: int = 40, max_lines_per_cue: int = 2,
    speaker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return _timed_cues(
        _chunk_text(text, max_chars_per_line, max_lines_per_cue),
        start_time, end_time, speaker,
    )


def wrap_cue_lines(text: str, max_chars_per_line: int = 40, max_lines: int = 2) -> str:
    chunks = _chunk_text(text, max_chars_per_line, max_lines)
    if len(chunks) > 1:
        raise ValueError("Text requires more than one cue to satisfy the requested limits")
    return chunks[0] if chunks else ""


def generate_subtitles_from_segments(
    segments: List[Dict[str, Any]], max_chars_per_line: int = 40,
    max_lines_per_cue: int = 2, use_translated: bool = False,
    bilingual: bool = False,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    previous_end = 0.0
    for segment in sorted(segments, key=lambda item: float(item.get("start", 0.0))):
        start = max(previous_end, float(segment.get("start", 0.0)))
        end = max(start + 0.001, float(segment.get("end", start + 2.0)))
        speaker = segment.get("speaker")
        original = str(segment.get("original_text", segment.get("text", ""))).strip()
        translated = str(segment.get("translated_text", "")).strip()
        if bilingual and original and translated:
            original_parts = _chunk_text(original, max_chars_per_line, 1)
            translated_parts = _chunk_text(translated, max_chars_per_line, 1)
            count = max(len(original_parts), len(translated_parts))
            original_parts = _partition_text(original, count, max_chars_per_line)
            translated_parts = _partition_text(translated, count, max_chars_per_line)
            chunks = [f"{target}\n{source}".strip() for target, source in zip(translated_parts, original_parts)]
            cues = _timed_cues(chunks, start, end, speaker)
        else:
            text = translated if use_translated and translated else original
            cues = split_text_into_cues(text, start, end, max_chars_per_line, max_lines_per_cue, speaker)
        if cues:
            previous_end = cues[-1]["end"]
            formatted.extend(cues)
    return formatted


def validate_subtitle_cues(
    cues: List[Dict[str, Any]], max_chars_per_line: int = 40,
    max_lines_per_cue: int = 2, max_chars_per_second: float = 20.0,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    previous_end = 0.0
    for index, cue in enumerate(cues, 1):
        start, end = float(cue["start"]), float(cue["end"])
        lines = str(cue.get("text", "")).splitlines() or [""]
        if end <= start:
            errors.append(f"Cue {index} has non-positive duration")
        if start < previous_end - 0.001:
            errors.append(f"Cue {index} overlaps the preceding cue")
        if len(lines) > max_lines_per_cue:
            errors.append(f"Cue {index} exceeds {max_lines_per_cue} lines")
        if any(len(line) > max_chars_per_line for line in lines):
            errors.append(f"Cue {index} exceeds {max_chars_per_line} characters per line")
        cps = len("".join(lines)) / max(0.001, end - start)
        if cps > max_chars_per_second:
            warnings.append(f"Cue {index} reads at {cps:.1f} characters/second")
        previous_end = max(previous_end, end)
    return {"valid": not errors, "errors": errors, "warnings": warnings, "cue_count": len(cues)}


def export_srt(cues: List[Dict[str, Any]]) -> str:
    blocks = []
    for index, cue in enumerate(cues, 1):
        speaker = f"[{cue['speaker']}] " if cue.get("speaker") else ""
        blocks.append(f"{index}\n{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n{speaker}{str(cue.get('text', '')).strip()}\n")
    return "\n".join(blocks)


def export_vtt(cues: List[Dict[str, Any]]) -> str:
    blocks = ["WEBVTT\n"]
    for cue in cues:
        speaker = f"<v {cue['speaker']}>" if cue.get("speaker") else ""
        blocks.append(f"{format_timestamp(cue['start'], True)} --> {format_timestamp(cue['end'], True)}\n{speaker}{str(cue.get('text', '')).strip()}\n")
    return "\n".join(blocks)
