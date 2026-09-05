#!/usr/bin/env python3
"""
Refined Local LLM test using Ollama /api/chat endpoint with proper role separation
and targeted executive-level prompting to match GenAI_Hackathon_Brainstorming_MOM.md.
"""

import os
import sys
import json
import re
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

JSON_PATH = r"F:\cohereX\outputs\WhatsApp Audio 2026-09-05 at 4.54.20 PM\WhatsApp Audio 2026-09-05 at 4.54.20 PM.json"
REFERENCE_MOM_PATH = r"C:\Users\omars\Documents\Codex\2026-09-05\activating-conda-environment-coherex-launching-streamlit\outputs\GenAI_Hackathon_Brainstorming_MOM.md"
OUTPUT_DIR = r"F:\cohereX\outputs\WhatsApp Audio 2026-09-05 at 4.54.20 PM"

def clean_repetitions(text: str) -> str:
    text = re.sub(r'([a-zA-Z\u0600-\u06FF])\1{3,}', r'\1', text)
    text = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'(\b\w+\s+\w+\b)(?:\s+\1){2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_seconds(s: float) -> str:
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"

print("📖 Loading transcript from JSON...")
with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

segments = data.get("segments", [])
transcript_lines = []
for seg in segments:
    st = seg.get("start", 0.0)
    et = seg.get("end", 0.0)
    text = clean_repetitions(seg.get("text", ""))
    if text:
        transcript_lines.append(f"[{format_seconds(st)} - {format_seconds(et)}] {text}")

full_transcript = "\n".join(transcript_lines)
word_count = len(full_transcript.split())
print(f"Loaded {len(transcript_lines)} segments ({word_count} words).")

# Check available Ollama models
req = urllib.request.Request("http://localhost:11434/api/tags")
with urllib.request.urlopen(req) as response:
    tags_data = json.loads(response.read().decode("utf-8"))
available_models = [m["name"] for m in tags_data.get("models", [])]
print(f"Available models: {available_models}")

selected_model = None
for candidate in ["qwen2.5:7b-instruct", "qwen2.5:7b", "deepseek-r1:7b"]:
    for m in available_models:
        if candidate in m:
            selected_model = m
            break
    if selected_model:
        break
if not selected_model:
    selected_model = available_models[0]

print(f"🤖 Selected model: {selected_model}")

SYSTEM_PROMPT = """You are a Principal Product Strategist and Executive Chief of Staff documenting an informal 27-minute brainstorming meeting conducted in Egyptian Arabic (Masri) and English.
The meeting is about designing a GenAI Hackathon project: an AI-powered conversational career discovery and guidance companion.

Your task is to write the definitive, comprehensive Minutes of Meeting (MOM).
You must write an in-depth, publication-grade strategic document that captures every detail discussed.

REQUIRED DOCUMENT STRUCTURE (follow these exact headings and depth):

# Minutes of Meeting: GenAI Hackathon Idea Brainstorming

## Meeting Overview
- Meeting purpose
- Primary focus
- Source duration (~26 minutes and 50 seconds)
- Meeting format (informal group brainstorming)
- Decision status

## Executive Summary
(Write 4 rich, detailed paragraphs explaining:
Paragraph 1: The core platform concept — an adaptive conversational AI for career guidance targeting people at education/career crossroads.
Paragraph 2: How it works vs rigid questionnaires — exploring personality, interests, skills, strengths through natural dialogue, recommending realistic career paths and explaining 'why'.
Paragraph 3: The core value proposition — reducing decision paralysis, building confidence to take a practical first step (not finding a single lifelong job).
Paragraph 4: Scope triage — acknowledging larger ideas (personalized teaching, continuous mentoring) while establishing the realistic hackathon MVP focus.)

## Problem Statement
(Describe the widespread dilemma of educational and career uncertainty among students and graduates, followed by a bulleted list analyzing the specific shortcomings of traditional assessments like 16personalities and static questionnaires: boring, rigid, multiple-choice limitations, lack of convincing reasoning, generic labels, no actionable next steps.)

## Proposed Solution
(Provide a numbered list of 8 specific core capabilities of the conversational AI career companion.)

## Target Users
(Break down the audience into distinct user personas: secondary school / Thanaweya Amma students, university students choosing a major/track, recent graduates, early-career professionals, career changers, freelancers.)

## Intended User Journey
(Provide a 9-step chronological user journey: Onboarding -> Adaptive Assessment -> Profile Creation -> Career Recommendations -> Explainable Reasoning -> Realistic Job Preview -> Decision Support -> Initial Roadmap -> Saved Profile & Continuity.)

## Proposed Features
### Core MVP Features (bulleted list of essential features for the demo)
### Optional Prototype Enhancements (stretch goals discussed)
### Future Vision (long-term platform ambitions)

## Potential Differentiators
(Detailed bolded bullet points explaining how this solution stands out from traditional chatbots and tests.)

## Career Profile Data Discussed
(Specific data points each job profile should include: responsibilities, skills, education, remote work, salary/demand in Egypt, day-in-the-life videos, sample tasks.)

## Technical and Product Considerations
(Key architectural discussions: fine-tuning vs prompt engineering, video curation from YouTube vs generation, user feedback loops, localized market data.)

## Key Concerns and Differing Views
(Detail the specific debates between participants:
### Scope Risk
### Novelty and Value
### Effectiveness and Trust
### Assessment Length
### One Career Versus Several)

## Working Scope for the Hackathon Prototype
(The realistic 8-step journey that can actually be built and demonstrated at the hackathon.)

## Preliminary Decisions and Agreements
(Bulleted list of consensus points.)

## Open Questions
(Detailed list of unresolved questions.)

## Recommended Next Steps
(10 numbered, concrete action items.)

## Draft One-Sentence Pitch
(A single bolded, compelling sentence summarizing the product.)

## Draft Problem Statement for the Hackathon
(A crisp, formal 3-4 sentence problem statement.)

GUIDELINES:
- Write strictly in English.
- Translate all Egyptian dialect discussions into clear, precise professional terminology.
- Be exhaustive, insightful, and structured. Do not skip sections or write placeholders.
- Base every single point strictly on the provided transcript."""

USER_MSG = f"""Here is the complete transcript of the 26-minute brainstorming meeting:

=== TRANSCRIPT ===
{full_transcript}
=== END TRANSCRIPT ===

Now produce the complete, exhaustive Minutes of Meeting following the required structure and depth."""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_MSG}
]

payload = {
    "model": selected_model,
    "messages": messages,
    "stream": False,
    "options": {
        "num_ctx": 16384,
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": 4096
    }
}

print("🚀 Sending request to Ollama /api/chat with proper ChatML role separation...")
t0 = time.time()

data_bytes = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=data_bytes,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=360) as response:
        res = json.loads(response.read().decode("utf-8"))
        elapsed = time.time() - t0
        
        msg = res.get("message", {})
        content = msg.get("content", "")
        
        # Clean any thinking tags
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        eval_count = res.get("eval_count", 0)
        eval_duration = res.get("eval_duration", 1) / 1e9
        speed = eval_count / eval_duration if eval_duration > 0 else 0
        
        print(f"✅ Generated in {elapsed:.1f}s ({eval_count} tokens, {speed:.1f} tok/s)!")
        print(f"Content length: {len(clean_content)} characters, {len(clean_content.split())} words.")
        
        out_path = os.path.join(OUTPUT_DIR, f"Local_LLM_{selected_model.replace(':', '_')}_Refined_MOM.md")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(clean_content)
        print(f"💾 Saved to: {out_path}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
