#!/usr/bin/env python3
r"""
CohereX CLI Subtitle & Transcription Tool
Usage:
  python scripts/cli_transcribe.py "path/to/movie.mp4" --lang ar --translate en --bilingual
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure root is in path
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
from coherex.subtitles import generate_subtitles_from_segments, export_srt, export_vtt


def main():
    parser = argparse.ArgumentParser(description="CohereX Broadcast Subtitle & ASR CLI")
    parser.add_argument("media_file", help="Path to input audio or video file")
    parser.add_argument("-l", "--lang", default="ar", help="Audio language (ar, en, es, fr, auto, etc.)")
    parser.add_argument("-t", "--translate", help="Target translation language (e.g. en, es, fr)")
    parser.add_argument("--bilingual", action="store_true", help="Generate dual-track bilingual subtitles")
    parser.add_argument("-o", "--output-dir", default="outputs", help="Output directory for generated subtitles")
    parser.add_argument("--max-chars", type=int, default=38, help="Max characters per subtitle line (default: 38)")
    parser.add_argument("--max-lines", type=int, default=2, help="Max lines per subtitle cue (default: 2)")
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

    print(f"🎬 Processing: {media_path.name}")
    print(f"🧠 Selected Model: {model_id} (Language: {args.lang})")

    # Step 1: Extract audio if video
    audio_path = extract_audio_from_video(media_path)
    audio_array = coherex.load_audio(str(audio_path))

    # Step 2: Transcribe
    print("🎙️ Transcribing speech...")
    model = coherex.load_model(
        model_name=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size,
        language=(None if args.lang == "auto" else args.lang)
    )
    result = model.transcribe(audio_array, batch_size=args.batch_size)
    detected_lang = result.get("language", args.lang)

    # Step 3: Translation (Optional)
    if args.translate:
        print(f"🌍 Translating subtitles into [{args.translate.upper()}]...")
    # Create dedicated subfolder per media file
    run_dir = out_dir / stem_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Export broadcast-timed subtitle cues
    print(f"⏱️ Formatting subtitle cues (max {args.max_chars} chars/line, max {args.max_lines} lines)...")
    segments = result.get("segments", [])

    cues_orig = generate_subtitles_from_segments(segments, args.max_chars, args.max_lines, use_translated=False)
    (run_dir / f"{stem_name}_{detected_lang}.srt").write_text(export_srt(cues_orig), encoding="utf-8")
    (run_dir / f"{stem_name}_{detected_lang}.vtt").write_text(export_vtt(cues_orig), encoding="utf-8")

    if args.translate:
        cues_trans = generate_subtitles_from_segments(segments, args.max_chars, args.max_lines, use_translated=True)
        (run_dir / f"{stem_name}_{args.translate}.srt").write_text(export_srt(cues_trans), encoding="utf-8")
        (run_dir / f"{stem_name}_{args.translate}.vtt").write_text(export_vtt(cues_trans), encoding="utf-8")

        if args.bilingual:
            cues_bi = generate_subtitles_from_segments(segments, args.max_chars, args.max_lines, use_translated=True, bilingual=True)
            (run_dir / f"{stem_name}_bilingual.srt").write_text(export_srt(cues_bi), encoding="utf-8")

    # Step 5: Save JSON
    with open(run_dir / f"{stem_name}.json", "w", encoding="utf-8") as jf:
        json.dump(result, jf, indent=2, ensure_ascii=False)

    print(f"\n🎉 Subtitle generation complete! Files saved to: {run_dir}")


if __name__ == "__main__":
    main()
