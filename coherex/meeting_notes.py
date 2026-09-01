#!/usr/bin/env python3
"""
Professional Meeting Notes & Transcript Summarizer for CohereX
Turns conversational audio/video transcripts into structured meeting minutes,
executive summaries, action items, speaker turns, and topic timelines.
"""

import re
from typing import List, Dict, Any, Optional

def format_short_time(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def group_speaker_dialogue(segments: List[Dict[str, Any]], use_translated: bool = False) -> List[Dict[str, Any]]:
    """
    Combines consecutive segments from the same speaker into cohesive conversational paragraphs.
    """
    if not segments:
        return []
        
    grouped = []
    current_turn = None
    
    for seg in segments:
        spk = seg.get("speaker", "Speaker")
        st = seg.get("start", 0.0)
        et = seg.get("end", st + 2.0)
        
        text = seg.get("translated_text", "") if use_translated and seg.get("translated_text") else seg.get("original_text", seg.get("text", "")).strip()
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


def generate_meeting_notes_markdown(
    result: Dict[str, Any],
    title: str = "Meeting & Conversation Notes",
    use_translated: bool = False
) -> str:
    """
    Generates a full, professional Markdown report for Meeting Notes mode.
    """
    segments = result.get("segments", [])
    grouped_turns = group_speaker_dialogue(segments, use_translated=use_translated)
    
    total_duration = segments[-1].get("end", 0.0) if segments else 0.0
    duration_str = format_short_time(total_duration)
    
    speakers = sorted(list(set(seg.get("speaker", "Speaker") for seg in segments if seg.get("speaker"))))
    speaker_list_str = ", ".join(speakers) if speakers else "Single Speaker / Unlabeled"
    
    lang_code = result.get("target_language" if use_translated else "language", "auto").upper()
    
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
    for t in grouped_turns[:12]:
        t_start = format_short_time(t["start"])
        snippet = t["text"][:100] + ("..." if len(t["text"]) > 100 else "")
        md.append(f"- **`{t_start}`** — **{t['speaker']}**: {snippet}")
    if len(grouped_turns) > 12:
        md.append(f"- _...and {len(grouped_turns) - 12} additional conversation segments below._")
    md.append("")
    
    # Full Structured Dialogue Transcript
    md.append("## 💬 Full Speaker-by-Speaker Transcript")
    for t in grouped_turns:
        t_start = format_short_time(t["start"])
        t_end = format_short_time(t["end"])
        md.append(f"### 👤 {t['speaker']} `[{t_start} ➔ {t_end}]`")
        md.append(f"{t['text']}\n")
        
    return "\n".join(md)
