#!/usr/bin/env python3
"""
Professional Subtitle Generator and Segment Splitter for CohereX
Converts raw ASR segments into standard broadcast-quality subtitle cues (SRT/VTT)
with customizable line length, line count, and natural sentence pacing.
"""

import re
import math
from typing import List, Dict, Any, Optional

def format_timestamp(seconds: float, is_vtt: bool = False) -> str:
    """Formats seconds into SRT (00:00:00,000) or VTT (00:00:00.000) timestamp."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    
    minutes = total_ms // 60_000
    total_ms %= 60_000
    
    secs = total_ms // 1_000
    ms = total_ms % 1_000
    
    sep = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def split_text_into_cues(
    text: str,
    start_time: float,
    end_time: float,
    max_chars_per_line: int = 40,
    max_lines_per_cue: int = 2,
    speaker: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Splits a single long segment into multiple small, natural subtitle cues
    based on punctuation boundaries, word count, and reading speed.
    """
    text = text.strip()
    if not text:
        return []
    
    duration = max(0.5, end_time - start_time)
    
    # Check if segment is already short enough for a single cue
    max_cue_chars = max_chars_per_line * max_lines_per_cue
    if len(text) <= max_cue_chars and duration <= 6.0:
        wrapped_text = wrap_cue_lines(text, max_chars_per_line, max_lines_per_cue)
        return [{
            "start": start_time,
            "end": end_time,
            "text": wrapped_text,
            "speaker": speaker
        }]
    
    # Split text into natural sentence / clause clauses
    clause_delimiters = r'([.!؟?،,;:\n]+)'
    parts = re.split(clause_delimiters, text)
    
    clauses = []
    current_clause = ""
    for p in parts:
        if re.match(clause_delimiters, p):
            current_clause += p
            if len(current_clause.strip()) > 0:
                clauses.append(current_clause.strip())
                current_clause = ""
        else:
            if current_clause:
                current_clause += " " + p
            else:
                current_clause = p
    if current_clause.strip():
        clauses.append(current_clause.strip())
        
    # Group clauses into chunks that fit within max_cue_chars
    chunks = []
    temp_chunk = ""
    for clause in clauses:
        candidate = f"{temp_chunk} {clause}".strip() if temp_chunk else clause
        if len(candidate) <= max_cue_chars:
            temp_chunk = candidate
        else:
            if temp_chunk:
                chunks.append(temp_chunk)
            # If a single clause is longer than max_cue_chars, split by words
            if len(clause) > max_cue_chars:
                words = clause.split()
                w_temp = ""
                for w in words:
                    cand_w = f"{w_temp} {w}".strip() if w_temp else w
                    if len(cand_w) <= max_cue_chars:
                        w_temp = cand_w
                    else:
                        if w_temp:
                            chunks.append(w_temp)
                        w_temp = w
                if w_temp:
                    temp_chunk = w_temp
                else:
                    temp_chunk = ""
            else:
                temp_chunk = clause
    if temp_chunk:
        chunks.append(temp_chunk)
        
    if not chunks:
        chunks = [text]
        
    # Distribute duration proportionally across chunks based on character length
    total_len = sum(len(c) for c in chunks)
    cues = []
    curr_start = start_time
    
    for i, c in enumerate(chunks):
        c_len = len(c)
        c_fraction = c_len / total_len if total_len > 0 else (1.0 / len(chunks))
        c_dur = duration * c_fraction
        
        c_dur = max(1.2, c_dur)
        curr_end = min(end_time, curr_start + c_dur)
        
        if i == len(chunks) - 1:
            curr_end = end_time
            
        wrapped = wrap_cue_lines(c, max_chars_per_line, max_lines_per_cue)
        cues.append({
            "start": round(curr_start, 3),
            "end": round(curr_end, 3),
            "text": wrapped,
            "speaker": speaker
        })
        curr_start = curr_end
        
    return cues


def wrap_cue_lines(text: str, max_chars_per_line: int = 40, max_lines: int = 2) -> str:
    """Wraps text into 1 or 2 balanced subtitle lines."""
    words = text.split()
    if not words:
        return text
    
    if len(text) <= max_chars_per_line:
        return text
        
    lines = []
    cur_line = ""
    for w in words:
        cand = f"{cur_line} {w}".strip() if cur_line else w
        if len(cand) <= max_chars_per_line or not cur_line:
            cur_line = cand
        else:
            lines.append(cur_line)
            cur_line = w
            if len(lines) >= max_lines - 1:
                break
                
    if cur_line:
        remaining_idx = len(" ".join(lines + [cur_line]).split())
        rest_words = words[remaining_idx:]
        if rest_words:
            cur_line = f"{cur_line} {' '.join(rest_words)}"
        lines.append(cur_line)
        
    return "\n".join(lines[:max_lines])


def generate_subtitles_from_segments(
    segments: List[Dict[str, Any]],
    max_chars_per_line: int = 40,
    max_lines_per_cue: int = 2,
    use_translated: bool = False,
    bilingual: bool = False
) -> List[Dict[str, Any]]:
    """
    Takes raw ASR segments and returns tight, broadcast-quality subtitle cues.
    """
    formatted_cues = []
    
    for seg in segments:
        st = seg.get("start", 0.0)
        et = seg.get("end", st + 2.0)
        spk = seg.get("speaker")
        
        orig_text = seg.get("original_text", seg.get("text", "")).strip()
        trans_text = seg.get("translated_text", "").strip()
        
        if bilingual and trans_text and orig_text:
            trans_cues = split_text_into_cues(trans_text, st, et, max_chars_per_line, 1, spk)
            orig_cues = split_text_into_cues(orig_text, st, et, max_chars_per_line, 1, spk)
            
            for idx in range(max(len(trans_cues), len(orig_cues))):
                tc = trans_cues[min(idx, len(trans_cues)-1)]
                oc = orig_cues[min(idx, len(orig_cues)-1)]
                combined = f"{tc['text']}\n{oc['text']}"
                formatted_cues.append({
                    "start": min(tc["start"], oc["start"]),
                    "end": max(tc["end"], oc["end"]),
                    "text": combined,
                    "speaker": spk
                })
        else:
            chosen_text = trans_text if (use_translated and trans_text) else orig_text
            cues = split_text_into_cues(chosen_text, st, et, max_chars_per_line, max_lines_per_cue, spk)
            formatted_cues.extend(cues)
            
    return formatted_cues


def export_srt(cues: List[Dict[str, Any]]) -> str:
    """Exports formatted cues to standard SRT string."""
    lines = []
    for idx, cue in enumerate(cues, 1):
        st = format_timestamp(cue["start"], is_vtt=False)
        et = format_timestamp(cue["end"], is_vtt=False)
        spk_tag = f"[{cue['speaker']}] " if cue.get("speaker") else ""
        text = cue.get("text", "").strip()
        lines.append(f"{idx}\n{st} --> {et}\n{spk_tag}{text}\n")
    return "\n".join(lines)


def export_vtt(cues: List[Dict[str, Any]]) -> str:
    """Exports formatted cues to standard WebVTT string."""
    lines = ["WEBVTT\n"]
    for cue in cues:
        st = format_timestamp(cue["start"], is_vtt=True)
        et = format_timestamp(cue["end"], is_vtt=True)
        spk_tag = f"<v {cue['speaker']}>" if cue.get("speaker") else ""
        text = cue.get("text", "").strip()
        lines.append(f"{st} --> {et}\n{spk_tag}{text}\n")
    return "\n".join(lines)
