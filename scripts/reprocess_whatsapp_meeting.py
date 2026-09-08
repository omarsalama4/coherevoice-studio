#!/usr/bin/env python3
"""
Reprocess WhatsApp Meeting Audio with 4-speaker diarization and clean dual-track meeting notes
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Load environment
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            os.environ[k] = v
            if k.lower() in ["hf_key", "hf_token"]:
                os.environ["HF_TOKEN"] = v

import torch
import coherex
from coherex.extract_audio import extract_audio_from_video
from coherex.translator import translate_result
from coherex.meeting_notes import generate_meeting_notes_markdown

parser = argparse.ArgumentParser(description="Reprocess a meeting with fixed speaker count")
parser.add_argument("media_file")
parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs"))
parser.add_argument("--speakers", type=int, default=4)
args = parser.parse_args()

input_media = Path(args.media_file).expanduser().resolve(strict=True)
out_dir = Path(args.output_dir).resolve() / input_media.stem
out_dir.mkdir(parents=True, exist_ok=True)
stem_name = input_media.stem

print(f"🎬 Processing: {input_media.name}")
print(f"📁 Output Directory: {out_dir}")

# Step 1: Extract Audio
print("🔊 Step 1: Extracting 16kHz mono audio...")
temporary_dir = Path(tempfile.mkdtemp(prefix="coherex_reprocess_"))
audio_path = extract_audio_from_video(input_media, output_path=temporary_dir / "normalized.wav")
audio_array = coherex.load_audio(str(audio_path))
print(f"✅ Audio duration: {len(audio_array)/16000:.1f}s")

# Step 2: Transcribe with Arabic Model
print("🎙️ Step 2: Transcribing speech with Cohere Transcribe Arabic...")
model = coherex.load_model(
    model_name="CohereLabs/cohere-transcribe-arabic-07-2026",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
    language="ar"
)
asr_result = model.transcribe(audio_array, batch_size=8)
print(f"✅ Recognized {len(asr_result.get('segments', []))} speech segments.")

# Step 3: Speaker Diarization with 4 speakers
print("👥 Step 3: Performing 4-speaker diarization...")
hf_token = os.environ.get("HF_TOKEN")
diarize_pipe = coherex.DiarizationPipeline(
    token=hf_token,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
diarize_segments = diarize_pipe(audio_array, num_speakers=args.speakers)
final_result = coherex.assign_word_speakers(diarize_segments, asr_result, fill_nearest=False)

detected_speakers = sorted(list(set(seg.get("speaker") for seg in final_result.get("segments", []) if seg.get("speaker"))))
print(f"✅ Diarization completed! Detected speakers: {detected_speakers}")

# Step 4: Full English Translation
print("🌍 Step 4: Translating meeting notes into English...")
translated_result = translate_result(final_result, target_lang="en", source_lang="ar")

# Step 5: Export Arabic and English Meeting Notes
print("📋 Step 5: Exporting clean meeting notes...")
notes_ar = generate_meeting_notes_markdown(
    final_result,
    title=f"حوار وملاحظات الاجتماع - {stem_name}",
    use_translated=False,
    allow_heuristic_fallback=False,
)
(out_dir / f"{stem_name}_meeting_notes_ar.md").write_text(notes_ar, encoding="utf-8")

client = coherex.get_llm_client()
if client is None:
    raise RuntimeError("English meeting-note translation requires a configured LLM provider")
notes_en = client.translate_meeting_notes(notes_ar, source_lang="ar", target_lang="en")
(out_dir / f"{stem_name}_meeting_notes_en.md").write_text(notes_en, encoding="utf-8")

(out_dir / f"{stem_name}_meeting_notes.txt").write_text(notes_en, encoding="utf-8")

with open(out_dir / f"{stem_name}.json", "w", encoding="utf-8") as jf:
    json.dump(translated_result, jf, indent=2, ensure_ascii=False)

print("\n🎉 ALL OUTPUTS SAVED:")
print(f"  📄 Arabic Notes:    {out_dir / f'{stem_name}_meeting_notes_ar.md'}")
print(f"  📄 English Notes:   {out_dir / f'{stem_name}_meeting_notes_en.md'}")
print(f"  📁 Full Run Folder: {out_dir}")

import shutil
shutil.rmtree(temporary_dir, ignore_errors=True)
