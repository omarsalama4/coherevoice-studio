#!/usr/bin/env python3
r"""
CohereX CLI Meeting Notes & Summarizer Tool
Usage:
  python scripts/cli_meeting_notes.py "path/to/meeting.mp4" --lang ar --translate en
"""

import os
import sys
import json
import argparse
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import coherex
from coherex.extract_audio import extract_audio_from_video
from coherex.translator import translate_result
from coherex.meeting_notes import generate_meeting_notes_markdown


def main():
    parser = argparse.ArgumentParser(description="CohereX Meeting Notes & Dialogue Summarizer CLI")
    parser.add_argument("media_file", help="Path to input audio or video file")
    parser.add_argument("-l", "--lang", default="ar", help="Audio language (ar, en, es, fr, auto, etc.)")
    parser.add_argument("-t", "--translate", help="Target translation language for notes (e.g. en)")
    parser.add_argument("-o", "--output-dir", default="outputs", help="Output directory for meeting notes")
    parser.add_argument("--batch-size", type=int, default=8, help="ASR batch size")

    args = parser.parse_args()

    media_path = Path(args.media_file).resolve()
    if not media_path.exists():
        print(f"❌ Error: File not found at {media_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = media_path.stem

    # Automatic Model Router
    if args.lang == "ar":
        model_id = "CohereLabs/cohere-transcribe-arabic-07-2026"
    else:
        model_id = "CohereLabs/cohere-transcribe-03-2026"

    print(f"📋 Generating Meeting Notes for: {media_path.name}")
    print(f"🧠 Selected Model: {model_id} (Language: {args.lang})")

    # Step 1: Extract audio if video
    temp_audio = out_dir / f".{stem_name}.{uuid.uuid4().hex}.wav"
    audio_path = extract_audio_from_video(media_path, output_path=temp_audio)
    try:
        audio_array = coherex.load_audio(str(audio_path))
    finally:
        Path(audio_path).unlink(missing_ok=True)

    # Step 2: Transcribe
    print("🎙️ Transcribing speech...")
    model = coherex.load_model(
        model_name=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size,
        language=None if args.lang == "auto" else args.lang,
    )
    detected_lang = coherex.detect_language(model, audio_array) if args.lang == "auto" else args.lang
    result = model.transcribe(audio_array, language=detected_lang, batch_size=args.batch_size)

    # Step 3: Diarization (Optional)
    token = os.environ.get("HF_TOKEN")
    if token:
        print("👥 Assigning speaker labels...")
        try:
            diarize_pipe = coherex.DiarizationPipeline(use_auth_token=token, device="cuda" if torch.cuda.is_available() else "cpu")
            diarize_segs = diarize_pipe(audio_array)
            result = coherex.assign_word_speakers(diarize_segs, result)
        except Exception as e:
            print(f"ℹ️ Diarization skipped: {e}")

    # Step 4: Translation (Optional)
    source_result = result
    if args.translate:
        print(f"🌍 Translating meeting notes into [{args.translate.upper()}]...")
        result = translate_result(result, target_lang=args.translate, source_lang=detected_lang)

    # Create dedicated subfolder per media file
    run_dir = out_dir / stem_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: Export structured meeting notes
    notes_md = generate_meeting_notes_markdown(
        source_result,
        title=f"Meeting Notes - {stem_name}",
        allow_heuristic_fallback=False,
    )
    if args.translate:
        client = coherex.get_llm_client()
        if client is None:
            raise RuntimeError("Meeting-note translation requires a configured LLM provider")
        notes_md = client.translate_meeting_notes(notes_md, source_lang=detected_lang, target_lang=args.translate)
    (run_dir / f"{stem_name}_meeting_notes.md").write_text(notes_md, encoding="utf-8")
    (run_dir / f"{stem_name}_meeting_notes.txt").write_text(notes_md, encoding="utf-8")

    # Step 6: Save JSON
    with open(run_dir / f"{stem_name}.json", "w", encoding="utf-8") as jf:
        json.dump(result, jf, indent=2, ensure_ascii=False)

    print(f"\n🎉 Meeting notes successfully generated! Saved to: {run_dir}")


if __name__ == "__main__":
    main()
