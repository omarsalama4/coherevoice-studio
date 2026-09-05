#!/usr/bin/env python3
"""
Test local open-source LLM on the WhatsApp Hackathon Brainstorming recording.
Generates an in-depth, executive-caliber MOM and validates it against the reference MOM.
"""

import os
import sys
import json
import re
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Path to the record transcript
JSON_PATH = r"F:\cohereX\outputs\WhatsApp Audio 2026-09-05 at 4.54.20 PM\WhatsApp Audio 2026-09-05 at 4.54.20 PM.json"
REFERENCE_MOM_PATH = r"C:\Users\omars\Documents\Codex\2026-09-05\activating-conda-environment-coherex-launching-streamlit\outputs\GenAI_Hackathon_Brainstorming_MOM.md"
OUTPUT_DIR = r"F:\cohereX\outputs\WhatsApp Audio 2026-09-05 at 4.54.20 PM"

def clean_repetitions(text: str) -> str:
    """Removes degenerate ASR loops like Faaaaaaaa... and Aaaaaaaa..."""
    text = re.sub(r'([a-zA-Z\u0600-\u06FF])\1{3,}', r'\1', text)
    text = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'(\b\w+\s+\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_seconds(s: float) -> str:
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"

print("📖 Step 1: Loading transcript from JSON...")
with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

segments = data.get("segments", [])
print(f"Loaded {len(segments)} segments.")

# Build cleaned transcript with timestamps
transcript_lines = []
for seg in segments:
    st = seg.get("start", 0.0)
    et = seg.get("end", 0.0)
    text = clean_repetitions(seg.get("text", ""))
    if text:
        transcript_lines.append(f"[{format_seconds(st)} - {format_seconds(et)}] {text}")

full_transcript = "\n".join(transcript_lines)
word_count = len(full_transcript.split())
print(f"Cleaned transcript: {len(transcript_lines)} lines, {word_count} words.")

# Determine active Ollama model
OLLAMA_URL = "http://localhost:11434/api/generate"

# Check available models
req = urllib.request.Request("http://localhost:11434/api/tags")
with urllib.request.urlopen(req) as response:
    tags_data = json.loads(response.read().decode("utf-8"))
available_models = [m["name"] for m in tags_data.get("models", [])]
print(f"Available Ollama models: {available_models}")

# Prefer qwen2.5:7b-instruct if ready, otherwise deepseek-r1:7b
selected_model = None
for candidate in ["qwen2.5:7b-instruct", "qwen2.5:7b", "deepseek-r1:7b"]:
    for m in available_models:
        if candidate in m:
            selected_model = m
            break
    if selected_model:
        break

if not selected_model:
    selected_model = available_models[0] if available_models else "deepseek-r1:7b"

print(f"🤖 Step 2: Using local open-source LLM: '{selected_model}'")

SYSTEM_PROMPT = """You are a world-class Executive Chief of Staff, Senior Product Strategist, and Technical Documenter.
You are analyzing a 26-minute audio recording transcript of a product brainstorming and strategy meeting held in Colloquial Egyptian Arabic (عامية مصرية / Masri) mixed with English tech terminology.

Your mission is to produce comprehensive, exhaustively detailed, professional Minutes of Meeting (MOM) matching executive consulting standards.

CRITICAL DIRECTIVES:
1. THOROUGH SYNTHESIS: Extract all substantive discussions, architectural proposals, user problems, disagreements, and product ideas across the ENTIRE transcript. Do not produce a brief summary; produce a deep, structured strategic document.
2. EGYPTIAN DIALECT EXPERTISE: The participants speak Egyptian Arabic and Arabizi. Understand colloquial phrases, metaphors, and cultural context (e.g., 'مش عارف تروح فين', 'عايز تجيب فيه ايه', 'تكبر المشروع', 'مش بنخترع العجلة', 'عايز يخرج مقتنع', '16 personalities', 'day in the life videos'). Translate all ideas into polished, professional executive English.
3. STRUCTURE: Organize the MOM into the following rigorous sections:
   # Minutes of Meeting: [Descriptive Meeting Title]
   ## Meeting Overview (Purpose, Primary Focus, Source Duration, Meeting Format, Decision Status)
   ## Executive Summary (Detailed 3-4 paragraph narrative covering vision, conversational assessment concept, value proposition, and MVP scope)
   ## Problem Statement (Core dilemma of educational/career confusion + specific flaws of current tools like static questionnaires/16personalities)
   ## Proposed Solution (Numbered list of core solution pillars)
   ## Target Users (Detailed breakdown of user segments: high school, university students, fresh graduates, career switchers)
   ## Intended User Journey (Step-by-step user interaction flow from onboarding to roadmap)
   ## Proposed Features (Grouped into: Core MVP Features, Optional Prototype Enhancements, Future Vision)
   ## Potential Differentiators (Key advantages over traditional career tests and chatbots)
   ## Career Profile Data Discussed (What information each job profile should contain: responsibilities, skills, salary, remote work, day-in-the-life videos)
   ## Technical and Product Considerations (Feasibility, fine-tuning vs prompt engineering, video curation vs generation, evaluation metrics)
   ## Key Concerns and Differing Views (Debates between participants: scope risk vs novelty, effectiveness, trust, assessment duration)
   ## Working Scope for the Hackathon Prototype (Achievable demo flow for the hackathon)
   ## Preliminary Decisions and Agreements
   ## Open Questions (Unresolved items for future alignment)
   ## Recommended Next Steps (Numbered, actionable tasks)
   ## Draft One-Sentence Pitch
   ## Draft Problem Statement for the Hackathon
4. ACCURACY: Ground every insight in the transcript. Do not invent facts or participants not present in the discussion.
5. CLEAN OUTPUT: Output strictly valid Markdown. Do not wrap your response in markdown code blocks."""

USER_PROMPT = f"""Below is the complete transcript of the 26-minute GenAI Hackathon idea brainstorming session:

=== SOURCE TRANSCRIPT ===
{full_transcript}
=== END SOURCE TRANSCRIPT ===

Please generate the comprehensive Minutes of Meeting (MOM) following the required structure."""

payload = {
    "model": selected_model,
    "prompt": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}",
    "stream": False,
    "options": {
        "num_ctx": 16384,
        "temperature": 0.25,
        "top_p": 0.9,
        "num_predict": 4096
    }
}

print(f"🚀 Step 3: Sending prompt to local LLM ({word_count} words input, num_ctx=16384)...")
t0 = time.time()

data_bytes = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    OLLAMA_URL,
    data=data_bytes,
    headers={"Content-Type": "application/json"}
)

try:
    # 5 minute timeout for local GPU inference
    with urllib.request.urlopen(req, timeout=300) as response:
        result_json = json.loads(response.read().decode("utf-8"))
        elapsed = time.time() - t0
        raw_response = result_json.get("response", "")
        
        # If model includes <think> tags (like DeepSeek R1), clean them for the final document
        clean_mom = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        eval_count = result_json.get("eval_count", 0)
        eval_duration = result_json.get("eval_duration", 1) / 1e9
        speed = eval_count / eval_duration if eval_duration > 0 else 0
        
        print(f"✅ Local LLM completed in {elapsed:.1f}s ({eval_count} tokens generated, ~{speed:.1f} tok/s)!")
        
        # Save output
        out_path = os.path.join(OUTPUT_DIR, f"Local_LLM_{selected_model.replace(':', '_')}_MOM.md")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(clean_mom)
        print(f"💾 Saved generated MOM to: {out_path} ({len(clean_mom)} characters)")
        
except Exception as e:
    print(f"❌ Error calling local LLM: {e}")
    sys.exit(1)
