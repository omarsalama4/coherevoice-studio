#!/usr/bin/env python3
"""
CohereX Studio - Production UI for ASR Transcription, Subtitles & Meeting Notes
Features:
- Dual Modes: 🎬 Professional Subtitles vs 📋 Meeting Notes & Summaries
- Automatic Model Routing: Specialized Arabic Dialect Model vs Multilingual Base Model
- Tight, Broadcast-Quality Subtitle Cues (No large screens of text)
- Dynamic Neural Translation (Not word-by-word)
- Direct Auto-Save to Disk for all outputs (.srt, .vtt, .txt, .md, .json)
"""

import os
import sys
import time
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
import torch

# Ensure current directory is on python sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Output directory
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def load_hf_token_from_env() -> str:
    """Auto-detects Hugging Face token from environment or .env file."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("hf_key") or os.environ.get("HF_KEY")
    if not token and (BASE_DIR / ".env").exists():
        try:
            with open(BASE_DIR / ".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip().lower()
                        v = v.strip().strip("'\"")
                        if k in ["hf_token", "hf_key", "huggingface_token", "hf_api_key"]:
                            token = v
                            break
        except Exception:
            pass
    if token:
        os.environ["HF_TOKEN"] = token
    return token or ""

import coherex
from coherex.utils import format_timestamp
from coherex.extract_audio import extract_audio_from_video, get_media_info
from coherex.translator import translate_result, POPULAR_LANGUAGES
from coherex.subtitles import generate_subtitles_from_segments, export_srt, export_vtt
from coherex.meeting_notes import generate_meeting_notes_markdown, group_speaker_dialogue

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="CohereVoice Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Brand Banner */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        background: linear-gradient(135deg, #181926 0%, #24273A 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }
    .brand-icon {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #6C5CE7, #A29BFE);
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.35);
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F4F4F5;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        color: #9CA3AF;
        font-size: 0.92rem;
        margin-top: 0.2rem;
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-green {
        background: rgba(34, 197, 94, 0.12);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .pill-blue {
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Mode Selection Banner */
    .mode-card {
        background: #1E1E2E;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Output banner */
    .save-banner {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    .save-path {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #A7F3D0;
        word-break: break-all;
    }

    /* Subtitle Card Preview */
    .sub-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6C5CE7;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.6rem;
        border-radius: 4px 8px 8px 4px;
    }
    .sub-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #9CA3AF;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        font-size: 0.95rem;
        color: #F3F4F6;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Top Brand Header
st.markdown(
    """
    <div class="brand-header">
        <div class="brand-icon">🎙️</div>
        <div>
            <h1 class="brand-title">CohereVoice Studio</h1>
            <div class="brand-subtitle">AI-Powered Speech Recognition, Broadcast Subtitles & Meeting Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# MODEL CACHE & PIPELINE
# ==============================================================================
@st.cache_resource(show_spinner="Loading Cohere ASR model into VRAM...")
def get_cohere_model(model_name: str, language_code: Optional[str] = None, batch_size: int = 8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_index = 0 if device == "cuda" else 0
    compute_type = "float16" if device == "cuda" else "float32"

    asr_kwargs = {
        "model_name": model_name,
        "device": device,
        "device_index": device_index,
        "compute_type": compute_type,
        "batch_size": batch_size,
    }
    if language_code and language_code != "auto":
        asr_kwargs["language"] = language_code

    return coherex.load_model(**asr_kwargs)


@st.cache_resource(show_spinner="Loading alignment model...")
def get_align_model(language_code: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return coherex.load_align_model(language_code=language_code, device=device)


@st.cache_resource(show_spinner="Loading speaker diarization model...")
def get_diarize_pipeline(hf_token: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return coherex.DiarizationPipeline(use_auth_token=hf_token, device=device)


def open_folder(folder_path: Path):
    """Opens output folder in file explorer."""
    try:
        if sys.platform == "win32":
            os.startfile(str(folder_path))
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(folder_path)])
    except Exception as e:
        st.warning(f"Could not open directory: {e}")


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.subheader("⚙️ System & Engine")
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    if cuda_available:
        st.markdown(f'<span class="status-pill pill-green">🟢 GPU: {gpu_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-blue">ℹ️ Running on CPU</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Audio Language Selection
    LANG_MAP = {
        "Arabic (Egyptian & Dialects)": "ar",
        "Auto-Detect Language": "auto",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Polish": "pl",
        "Greek": "el",
        "Japanese": "ja",
        "Chinese": "zh",
        "Vietnamese": "vi",
        "Korean": "ko",
    }
    selected_lang_name = st.selectbox(
        "Spoken Audio Language",
        list(LANG_MAP.keys()),
        index=0,
        help="Choose Arabic for Egyptian/Arab media to automatically route to the finetuned dialect model."
    )
    lang_code = LANG_MAP[selected_lang_name]

    # Automatic Model Selector
    if lang_code == "ar":
        active_model_id = "CohereLabs/cohere-transcribe-arabic-07-2026"
        model_desc = "Cohere Transcribe Arabic (Finetuned for Dialects & Slang)"
    else:
        active_model_id = "CohereLabs/cohere-transcribe-03-2026"
        model_desc = "Cohere Transcribe Base (14 Languages Multilingual)"

    st.info(f"🧠 **Active ASR Model:**\n`{model_desc}`")

    # Hugging Face Auth Token
    default_hf_token = load_hf_token_from_env()
    hf_token = st.text_input(
        "Hugging Face Token",
        value=default_hf_token,
        type="password",
        help="Loaded automatically from .env (hf_key)."
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    st.markdown("---")
    st.caption(f"📁 **Auto-Save Output Folder:**\n`{OUTPUTS_DIR}`")
    if st.button("📂 Open Outputs Folder", use_container_width=True):
        open_folder(OUTPUTS_DIR)

    # Advanced Settings (Collapsed)
    with st.expander("🛠️ Advanced Settings", expanded=False):
        batch_size = st.slider("ASR Batch Size", 1, 32, 8)
        vad_onset = st.slider("VAD Speech Sensitivity", 0.1, 0.9, 0.45, 0.05)
        vad_offset = st.slider("VAD Silence Release", 0.1, 0.9, 0.35, 0.05)


# ==============================================================================
# MAIN PAGE: 1. UPLOAD MEDIA & 2. TRANSCRIPTION PIPELINE MODE
# ==============================================================================
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    with st.container(border=True):
        st.subheader("📁 1. Media Input")
        upload_type = st.radio("Source:", ["Upload File", "Local File Path"], horizontal=True)

        media_path = None
        media_display_name = ""

        if upload_type == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Video or Audio (MP4, MKV, MP3, WAV, M4A, etc.)",
                type=["mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "m4a", "flac", "ogg", "aac"]
            )
            if uploaded_file is not None:
                media_display_name = uploaded_file.name
                temp_dir = Path(tempfile.gettempdir()) / "coherex_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_media_path = temp_dir / uploaded_file.name
                with open(temp_media_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                media_path = str(temp_media_path)
                st.success(f"✅ Loaded: `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.1f} MB)")
        else:
            default_path = r"C:\Users\omars\Downloads\videoplayback.mp4"
            local_path_str = st.text_input("Enter exact file path on your computer:", value=default_path)
            if local_path_str and Path(local_path_str).exists():
                media_path = local_path_str
                media_display_name = Path(local_path_str).name
                size_mb = Path(local_path_str).stat().st_size / (1024 * 1024)
                st.success(f"✅ Found: `{media_display_name}` ({size_mb:.1f} MB)")
            elif local_path_str:
                st.error("❌ File not found. Please verify the path.")

with col_right:
    with st.container(border=True):
        st.subheader("🎯 2. Select Transcription Output Mode")
        
        pipeline_mode = st.radio(
            "Choose your objective:",
            ["🎬 Subtitles & Closed Captions", "📋 Meeting Notes & Conversation Summary"],
            index=0,
            help="Subtitles creates short, readable, broadcast-timed subtitle cues. Meeting Notes creates structured summaries and action items."
        )

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        # Mode Specific Settings
        if pipeline_mode.startswith("🎬"):
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                max_chars_line = st.slider("Max Characters / Line", 25, 60, 38, help="Standard broadcast is 35-42 characters per line.")
            with sub_col2:
                max_lines_cue = st.slider("Max Lines / Subtitle", 1, 3, 2)

            # Translation Option for Subtitles
            enable_trans = st.checkbox("🌍 Translate Subtitles into Another Language", value=True)
            target_lang_code = "en"
            is_bilingual = False
            if enable_trans:
                tc1, tc2 = st.columns([3, 2])
                with tc1:
                    t_name = st.selectbox("Target Subtitle Language", list(POPULAR_LANGUAGES.keys()), index=0)
                    target_lang_code = POPULAR_LANGUAGES[t_name]
                with tc2:
                    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                    is_bilingual = st.checkbox("Dual-Track Bilingual (EN + AR)", value=True)
        else:
            # Meeting Notes Settings
            st.caption("Generates Executive Overview, Speaker Discussion Turns, and Action Items.")
            enable_trans = st.checkbox("🌍 Translate Meeting Notes into Another Language", value=False)
            target_lang_code = "en"
            is_bilingual = False
            if enable_trans:
                t_name = st.selectbox("Target Notes Language", list(POPULAR_LANGUAGES.keys()), index=0)
                target_lang_code = POPULAR_LANGUAGES[t_name]

        # Word alignment & Diarization
        enable_diarization = st.checkbox("👥 Identify & Label Speakers (Diarization)", value=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        start_btn = st.button(
            "🚀 Run Pipeline",
            type="primary",
            use_container_width=True,
            disabled=(media_path is None)
        )


# ==============================================================================
# EXECUTION WORKFLOW
# ==============================================================================
if start_btn and media_path:
    stage_status = st.empty()
    overall_progress = st.progress(0.0)
    live_detail_box = st.empty()

    try:
        t0 = time.time()
        
        # Step 1: Audio Extraction
        stage_status.info("🔊 **Step 1/5: Extracting and normalizing 16kHz mono audio...**")
        overall_progress.progress(0.10)
        audio_extracted_path = extract_audio_from_video(media_path)
        audio_array = coherex.load_audio(str(audio_extracted_path))
        live_detail_box.caption(f"✅ Audio extracted: {len(audio_array)/16000:.1f} seconds duration.")

        # Step 2: Load Model & Transcribe
        stage_status.info(f"🎙️ **Step 2/5: Transcribing with {model_desc}...**")
        overall_progress.progress(0.30)
        asr_model = get_cohere_model(
            model_name=active_model_id,
            language_code=(None if lang_code == "auto" else lang_code),
            batch_size=batch_size
        )
        
        vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}
        asr_result = asr_model.transcribe(
            audio_array,
            batch_size=batch_size,
            vad_options=vad_options
        )
        detected_lang = asr_result.get("language", lang_code)
        live_detail_box.caption(f"✅ Recognized {len(asr_result.get('segments', []))} speech segments. Detected Language: [{detected_lang.upper()}].")

        # Step 3: Phoneme Alignment
        stage_status.info("⏱️ **Step 3/5: Computing word-level timing...**")
        overall_progress.progress(0.55)
        try:
            align_model, align_metadata = get_align_model(language_code=detected_lang)
            aligned_result = coherex.align(
                asr_result["segments"],
                align_model,
                align_metadata,
                audio_array,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            final_result = aligned_result
            live_detail_box.caption("✅ Phoneme alignment complete.")
        except Exception as align_err:
            final_result = asr_result
            live_detail_box.caption(f"ℹ️ Standard alignment used: {align_err}")

        # Step 4: Speaker Diarization
        if enable_diarization:
            stage_status.info("👥 **Step 4/5: Diarizing speakers...**")
            overall_progress.progress(0.75)
            try:
                diarize_pipe = get_diarize_pipeline(hf_token=hf_token)
                diarize_segments = diarize_pipe(audio_array)
                final_result = coherex.assign_word_speakers(diarize_segments, final_result)
                live_detail_box.caption("✅ Speaker labels assigned.")
            except Exception as diarize_err:
                live_detail_box.caption(f"ℹ️ Diarization note: {diarize_err}")

        # Step 5: Subtitle Formatting or Translation
        if enable_trans:
            stage_status.info(f"🌍 **Step 5/5: Translating into [{target_lang_code.upper()}]...**")
            overall_progress.progress(0.90)
            final_result = translate_result(
                final_result,
                target_lang=target_lang_code,
                source_lang=detected_lang,
                bilingual=is_bilingual
            )
            live_detail_box.caption(f"✅ Translated into {target_lang_code.upper()}.")

        overall_progress.progress(1.0)
        elapsed = time.time() - t0

        # Save to Disk in Dedicated Subfolder Named After Media File
        stem_name = Path(media_display_name).stem
        run_output_dir = OUTPUTS_DIR / stem_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        if pipeline_mode.startswith("🎬"):
            # Format tight subtitle cues
            cues_orig = generate_subtitles_from_segments(final_result.get("segments", []), max_chars_line, max_lines_cue, use_translated=False)
            cues_trans = generate_subtitles_from_segments(final_result.get("segments", []), max_chars_line, max_lines_cue, use_translated=True)
            cues_bilingual = generate_subtitles_from_segments(final_result.get("segments", []), max_chars_line, max_lines_cue, use_translated=True, bilingual=True)

            orig_srt = run_output_dir / f"{stem_name}_{detected_lang}.srt"
            orig_srt.write_text(export_srt(cues_orig), encoding="utf-8")
            saved_paths["orig_srt"] = orig_srt

            orig_vtt = run_output_dir / f"{stem_name}_{detected_lang}.vtt"
            orig_vtt.write_text(export_vtt(cues_orig), encoding="utf-8")
            saved_paths["orig_vtt"] = orig_vtt

            if enable_trans:
                trans_srt = run_output_dir / f"{stem_name}_{target_lang_code}.srt"
                trans_srt.write_text(export_srt(cues_trans), encoding="utf-8")
                saved_paths["trans_srt"] = trans_srt

                trans_vtt = run_output_dir / f"{stem_name}_{target_lang_code}.vtt"
                trans_vtt.write_text(export_vtt(cues_trans), encoding="utf-8")
                saved_paths["trans_vtt"] = trans_vtt

                bi_srt = run_output_dir / f"{stem_name}_bilingual.srt"
                bi_srt.write_text(export_srt(cues_bilingual), encoding="utf-8")
                saved_paths["bilingual_srt"] = bi_srt

            # Save full metadata JSON
            json_path = run_output_dir / f"{stem_name}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(final_result, jf, indent=2, ensure_ascii=False)
            saved_paths["json"] = json_path

        else:
            # Meeting Notes Mode
            notes_md = generate_meeting_notes_markdown(final_result, title=f"Meeting Notes - {stem_name}", use_translated=enable_trans)
            md_path = run_output_dir / f"{stem_name}_meeting_notes.md"
            md_path.write_text(notes_md, encoding="utf-8")
            saved_paths["notes_md"] = md_path

            txt_path = run_output_dir / f"{stem_name}_meeting_notes.txt"
            txt_path.write_text(notes_md, encoding="utf-8")
            saved_paths["notes_txt"] = txt_path

            json_path = run_output_dir / f"{stem_name}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(final_result, jf, indent=2, ensure_ascii=False)
            saved_paths["json"] = json_path

        st.session_state["last_result"] = final_result
        st.session_state["media_name"] = media_display_name
        st.session_state["saved_paths"] = saved_paths
        st.session_state["run_output_dir"] = run_output_dir
        st.session_state["pipeline_mode"] = pipeline_mode

        stage_status.success(f"🎉 **Pipeline Completed in {elapsed:.1f}s! All files saved to `{run_output_dir}`.**")
        live_detail_box.empty()

    except Exception as e:
        stage_status.error(f"❌ Error during execution: {str(e)}")
        overall_progress.progress(0.0)
        with st.expander("🔍 View Error Diagnostics", expanded=True):
            st.code(traceback.format_exc())


# ==============================================================================
# RESULTS STUDIO WORKSPACE
# ==============================================================================
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    media_name = st.session_state.get("media_name", "audio")
    saved_paths = st.session_state.get("saved_paths", {})
    run_dir = st.session_state.get("run_output_dir", OUTPUTS_DIR)
    mode = st.session_state.get("pipeline_mode", "Subtitles")

    st.markdown("---")
    st.subheader(f"📊 Results & Output Studio ({mode})")

    # Output Banner
    st.markdown(
        f"""
        <div class="save-banner">
            <div>
                <strong style="color: #4ADE80;">💾 Outputs Saved in Dedicated Run Folder:</strong>
                <div class="save-path">{run_dir}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("📂 Open Run Folder in Explorer", use_container_width=False):
        open_folder(run_dir)

    if mode.startswith("🎬"):
        tab_sub1, tab_sub2, tab_sub3, tab_sub4 = st.tabs(["📝 Subtitles Preview", "🔤 Arabic SRT", "🌍 English SRT", "📑 Bilingual SRT"])
        
        with tab_sub1:
            cues = generate_subtitles_from_segments(res.get("segments", []), 38, 2, use_translated=True, bilingual=("bilingual_srt" in saved_paths))
            st.caption(f"Displaying {len(cues)} tightly synchronized broadcast subtitle cues:")
            for c in cues[:25]:
                st.markdown(
                    f"""
                    <div class="sub-card">
                        <div class="sub-time">⏱️ {format_timestamp(c['start'])} ➔ {format_timestamp(c['end'])}</div>
                        <div class="sub-text">{c['text'].replace(chr(10), '<br>')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if len(cues) > 25:
                st.caption(f"_...and {len(cues)-25} more subtitle cues saved in the output file._")

        with tab_sub2:
            if "orig_srt" in saved_paths:
                st.text_area("Arabic SRT Content", saved_paths["orig_srt"].read_text(encoding="utf-8"), height=350)
                st.download_button("⬇️ Download Arabic SRT", saved_paths["orig_srt"].read_text(encoding="utf-8"), file_name=saved_paths["orig_srt"].name)

        with tab_sub3:
            if "trans_srt" in saved_paths:
                st.text_area("English SRT Content", saved_paths["trans_srt"].read_text(encoding="utf-8"), height=350)
                st.download_button("⬇️ Download English SRT", saved_paths["trans_srt"].read_text(encoding="utf-8"), file_name=saved_paths["trans_srt"].name)

        with tab_sub4:
            if "bilingual_srt" in saved_paths:
                st.text_area("Bilingual SRT Content", saved_paths["bilingual_srt"].read_text(encoding="utf-8"), height=350)
                st.download_button("⬇️ Download Bilingual SRT", saved_paths["bilingual_srt"].read_text(encoding="utf-8"), file_name=saved_paths["bilingual_srt"].name)

    else:
        # Meeting Notes View
        if "notes_md" in saved_paths:
            notes_content = saved_paths["notes_md"].read_text(encoding="utf-8")
            st.markdown(notes_content)
            st.download_button("⬇️ Download Meeting Notes (.md)", notes_content, file_name=saved_paths["notes_md"].name)
