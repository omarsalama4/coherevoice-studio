#!/usr/bin/env python3
"""
CohereX Studio - Production UI for ASR Transcription, Subtitles & Meeting Intelligence
Features:
- Dual Modes: 🎬 Professional Subtitles & Video Synopsis vs 📋 Adaptive Meeting Minutes (MOM)
- Multi-Engine LLM Intelligence:
    1. 🖥️ Local Open-Source LLM (Ollama: Qwen 2.5 / DeepSeek) — 100% Offline GPU Processing
    2. 🤖 OpenAI API (GPT-4o, GPT-4o-mini, Groq, vLLM)
    3. ☁️ Google Gemini API (Gemini 2.0 Flash)
    4. ⚡ Heuristic Rule-Based Fallback
- Automatic Ollama background management & one-click start
- Auto-detected speaker diarization for broadcast subtitles and conversations
- Fast cached rendering (zero UI lag on widget clicks)
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

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# ENVIRONMENT & KEY LOADERS
# ==============================================================================
def load_env_vars() -> Dict[str, str]:
    """Load key-value pairs from .env file into environment."""
    env_data = {}
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        env_data[k] = v
                        if k not in os.environ and v:
                            os.environ[k] = v
        except Exception:
            pass
    return env_data

ENV_VARS = load_env_vars()

def get_env_val(keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in os.environ and os.environ[k]:
            return os.environ[k]
        if k in ENV_VARS and ENV_VARS[k]:
            return ENV_VARS[k]
    return default


import coherex
from coherex.utils import format_timestamp
from coherex.extract_audio import extract_audio_from_video, get_media_info
from coherex.translator import translate_result, POPULAR_LANGUAGES
from coherex.subtitles import generate_subtitles_from_segments, export_srt, export_vtt
from coherex.meeting_notes import generate_meeting_notes_markdown, group_speaker_dialogue
from coherex.llm.ollama_client import ensure_ollama_running

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
    .pill-yellow {
        background: rgba(234, 179, 8, 0.12);
        color: #FACC15;
        border: 1px solid rgba(234, 179, 8, 0.25);
    }
    .pill-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }

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

    .llm-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .llm-badge-active {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .llm-badge-local {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .llm-badge-fallback {
        background: rgba(234, 179, 8, 0.12);
        color: #FACC15;
        border: 1px solid rgba(234, 179, 8, 0.25);
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
            <div class="brand-subtitle">AI-Powered Speech Recognition, Broadcast Subtitles & Adaptive Meeting Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# FAST CACHED BACKEND RESOLVER (Eliminates UI freezing)
# ==============================================================================
@st.cache_data(ttl=25)
def get_cached_backend_status(gemini_k: str, openai_k: str) -> Dict[str, Any]:
    """Cached non-blocking check of local Ollama, OpenAI, and Gemini backends."""
    from coherex.llm import get_llm_backend_info
    return get_llm_backend_info(gemini_api_key=gemini_k, openai_api_key=openai_k)


# Model caching
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
    st.subheader("⚙️ System & GPU")
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    if cuda_available:
        st.markdown(f'<span class="status-pill pill-green">🟢 GPU: {gpu_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-blue">ℹ️ Running on CPU</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Spoken Language Selection
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
        help="Choose Arabic for Egyptian/Arab media to route to the specialized dialect model."
    )
    lang_code = LANG_MAP[selected_lang_name]

    if lang_code == "ar":
        active_model_id = "CohereLabs/cohere-transcribe-arabic-07-2026"
        model_desc = "Cohere Transcribe Arabic (Finetuned for Dialects & Slang)"
    else:
        active_model_id = "CohereLabs/cohere-transcribe-03-2026"
        model_desc = "Cohere Transcribe Base (14 Languages Multilingual)"

    st.info(f"🧠 **ASR Speech Engine:**\n`{model_desc}`")

    # ==========================================================================
    # LLM INTELLIGENCE ENGINE SECTION
    # ==========================================================================
    st.markdown("---")
    st.subheader("🧠 LLM Intelligence Engine")

    # Keys from env
    env_gemini = get_env_val(["GEMINI_API_KEY", "gemini_key", "google_api_key"])
    env_openai = get_env_val(["OPENAI_API_KEY", "openai_key"])
    env_openai_model = get_env_val(["OPENAI_MODEL"], "gpt-4o-mini")

    backend_info = get_cached_backend_status(env_gemini, env_openai)

    # Active status badge
    if backend_info["has_local"]:
        st.markdown(
            f'<span class="status-pill pill-green">🟢 Local LLM: {backend_info["local_model"]}</span>',
            unsafe_allow_html=True
        )
        st.caption("🔒 100% Offline GPU Processing — Zero API cost or data transfer.")
    elif backend_info["has_openai"]:
        st.markdown(
            f'<span class="status-pill pill-blue">🤖 Cloud: OpenAI ({backend_info["openai_model"]})</span>',
            unsafe_allow_html=True
        )
    elif backend_info["has_gemini"]:
        st.markdown(
            '<span class="status-pill pill-purple">☁️ Cloud: Google Gemini 2.0</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-pill pill-yellow">⚡ No LLM — Heuristic Mode</span>',
            unsafe_allow_html=True
        )
        st.caption("Start Ollama or add an OpenAI/Gemini API key below.")
        if st.button("🚀 Auto-Start Local Ollama", use_container_width=True):
            with st.spinner("Starting Ollama server in background..."):
                ok = ensure_ollama_running(timeout_secs=6)
                if ok:
                    st.success("✅ Ollama started!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Could not auto-start Ollama. Please check if Ollama is installed.")

    # Model selector if multiple local models exist
    if backend_info["has_local"] and len(backend_info["installed_models"]) > 1:
        from coherex.llm import get_local_llm_client
        loc_client = get_local_llm_client()
        cur_m = backend_info["local_model"]
        chosen_m = st.selectbox(
            "Local Model",
            backend_info["installed_models"],
            index=backend_info["installed_models"].index(cur_m) if cur_m in backend_info["installed_models"] else 0
        )
        if loc_client:
            loc_client.set_model(chosen_m)

    # Collapsible API Settings
    with st.expander("🔑 Cloud & API Credentials", expanded=False):
        default_hf = get_env_val(["HF_TOKEN", "hf_key", "hf_token"])
        hf_token = st.text_input("Hugging Face Token", value=default_hf, type="password")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        openai_key_input = st.text_input("OpenAI API Key (Optional)", value=env_openai, type="password")
        if openai_key_input:
            os.environ["OPENAI_API_KEY"] = openai_key_input

        openai_model_input = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "custom"],
            index=0
        )
        if openai_model_input != "custom":
            os.environ["OPENAI_MODEL"] = openai_model_input

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        gemini_key_input = st.text_input("Gemini API Key (Optional)", value=env_gemini, type="password")
        if gemini_key_input:
            os.environ["GEMINI_API_KEY"] = gemini_key_input

        if st.button("🔄 Refresh Engines", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    llm_available = backend_info["any_available"]

    st.markdown("---")
    st.caption(f"📁 **Auto-Save Folder:**\n`{OUTPUTS_DIR}`")
    if st.button("📂 Open Outputs Folder", use_container_width=True):
        open_folder(OUTPUTS_DIR)

    with st.expander("🛠️ ASR Advanced Tuning", expanded=False):
        batch_size = st.slider("ASR Batch Size", 1, 32, 8)
        vad_onset = st.slider("VAD Speech Sensitivity", 0.1, 0.9, 0.45, 0.05)
        vad_offset = st.slider("VAD Silence Release", 0.1, 0.9, 0.35, 0.05)


# ==============================================================================
# MAIN PAGE: 1. MEDIA INPUT & 2. PIPELINE MODE
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
            default_path = r"C:\Users\omars\Downloads\WhatsApp Audio 2026-09-05 at 4.54.20 PM.mp4"
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
        st.subheader("🎯 2. Select Output Objective")
        
        pipeline_mode = st.radio(
            "Choose mode:",
            ["🎬 Videos & Subtitles (SRT/VTT)", "📋 Meeting Notes & Adaptive MOM"],
            index=0,
            help="Subtitles creates broadcast-timed cues and scene synopsis. Meeting Notes creates an adaptive executive report with decisions and action items."
        )

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        generate_video_synopsis = False

        if pipeline_mode.startswith("🎬"):
            # Subtitle Settings
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                max_chars_line = st.slider("Max Chars / Line", 25, 60, 38, help="Broadcast standard is 35-42 chars.")
            with sub_c2:
                max_lines_cue = st.slider("Max Lines / Cue", 1, 3, 2)

            # Translation Option
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
                    is_bilingual = st.checkbox("Dual-Track (EN + AR)", value=True)

            # Video Intelligence (Serving Videos with LLM)
            if llm_available:
                generate_video_synopsis = st.checkbox(
                    "🎬 Generate AI Video Synopsis & Scene Breakdown",
                    value=True,
                    help="Uses the LLM to generate an executive synopsis, chapter timeline, and key takeaways."
                )

        else:
            # Meeting Notes Settings
            if llm_available:
                b_badge = "llm-badge-local" if backend_info["has_local"] else "llm-badge-active"
                st.markdown(
                    f'<span class="llm-badge {b_badge}">🧠 Adaptive Executive MOM</span>'
                    '<span style="margin-left: 0.5rem; color: #9CA3AF; font-size: 0.82rem;">'
                    'Tailored to transcript • Executive Summary • Decisions • Action Items</span>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("Rule-based structured dialogue turns and highlights.")

            enable_trans = st.checkbox("🌍 Translate Meeting Notes into English", value=False)
            target_lang_code = "en"
            is_bilingual = False

        # Speaker Diarization — Auto-detected by default
        enable_diarization = st.checkbox("👥 Auto-Detect & Label Speakers (Diarization)", value=True)
        exact_speakers = None
        if enable_diarization:
            spk_c1, spk_c2 = st.columns([3, 2])
            with spk_c1:
                spk_mode = st.selectbox("Speaker Detection Mode", ["Auto-Detect (2-6 speakers)", "Exact Number of Speakers"], index=0)
            with spk_c2:
                if spk_mode == "Exact Number of Speakers":
                    exact_speakers = st.number_input("Speaker Count", min_value=1, max_value=12, value=4)
                else:
                    exact_speakers = None

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
    is_meeting_mode = pipeline_mode.startswith("📋")
    use_llm_pipeline = llm_available
    total_steps = 6 if (use_llm_pipeline and (is_meeting_mode or generate_video_synopsis)) else 5

    stage_status = st.empty()
    overall_progress = st.progress(0.0)
    live_detail_box = st.empty()

    try:
        t0 = time.time()
        
        # Step 1: Audio Extraction
        stage_status.info(f"🔊 **Step 1/{total_steps}: Extracting and normalizing 16kHz audio...**")
        overall_progress.progress(0.10)
        audio_extracted_path = extract_audio_from_video(media_path)
        audio_array = coherex.load_audio(str(audio_extracted_path))
        live_detail_box.caption(f"✅ Audio extracted: {len(audio_array)/16000:.1f}s duration.")

        # Step 2: ASR Transcription
        stage_status.info(f"🎙️ **Step 2/{total_steps}: Transcribing with {model_desc}...**")
        overall_progress.progress(0.25)
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
        live_detail_box.caption(f"✅ Recognized {len(asr_result.get('segments', []))} speech segments. Language: [{detected_lang.upper()}].")

        # Step 3: Word-Level Phoneme Alignment
        stage_status.info(f"⏱️ **Step 3/{total_steps}: Computing word-level timing...**")
        overall_progress.progress(0.45)
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
            stage_status.info(f"👥 **Step 4/{total_steps}: Diarizing speakers...**")
            overall_progress.progress(0.60)
            try:
                diarize_pipe = get_diarize_pipeline(hf_token=hf_token)
                diarize_kwargs = {}
                if exact_speakers:
                    diarize_kwargs["num_speakers"] = int(exact_speakers)
                else:
                    diarize_kwargs["min_speakers"] = 2
                    diarize_kwargs["max_speakers"] = 8
                    
                diarize_segments = diarize_pipe(audio_array, **diarize_kwargs)
                final_result = coherex.assign_word_speakers(diarize_segments, final_result, fill_nearest=True)
                detected_spk_count = len(set(seg.get("speaker") for seg in final_result.get("segments", []) if seg.get("speaker")))
                live_detail_box.caption(f"✅ Diarization identified {detected_spk_count} distinct speakers.")
            except Exception as diarize_err:
                live_detail_box.caption(f"ℹ️ Diarization note: {diarize_err}")

        # Step 5: Subtitle Translation
        if enable_trans:
            trans_engine = "🧠 AI-Powered" if use_llm_pipeline else "⚡ Fast"
            stage_status.info(f"🌍 **Step 5/{total_steps}: {trans_engine} Translation into [{target_lang_code.upper()}]...**")
            overall_progress.progress(0.75)
            final_result = translate_result(
                final_result,
                target_lang=target_lang_code,
                source_lang=detected_lang,
                bilingual=is_bilingual,
                use_llm=use_llm_pipeline,
                gemini_api_key=gemini_key_input,
            )
            live_detail_box.caption(f"✅ Translated into {target_lang_code.upper()}.")

        # Step 6: LLM Intelligence (MOM or Video Synopsis)
        video_synopsis_text = None
        if use_llm_pipeline:
            from coherex.llm import get_llm_client
            active_llm = get_llm_client(gemini_api_key=gemini_key_input, openai_api_key=openai_key_input)

            if is_meeting_mode:
                stage_status.info(f"🧠 **Step {total_steps}/{total_steps}: Generating Adaptive Meeting Minutes...**")
                overall_progress.progress(0.90)
                live_detail_box.caption("Analyzing full conversation, extracting decisions, action items, and strategic topics...")
            elif generate_video_synopsis and active_llm:
                stage_status.info(f"🎬 **Step {total_steps}/{total_steps}: Generating AI Video Synopsis & Scene Breakdown...**")
                overall_progress.progress(0.90)
                try:
                    all_text = " ".join(s.get("text", "") for s in final_result.get("segments", []))
                    duration_str = format_timestamp(final_result.get("segments", [])[-1].get("end", 0.0))
                    if hasattr(active_llm, "generate_video_synopsis"):
                        video_synopsis_text = active_llm.generate_video_synopsis(all_text, media_display_name, duration_str, detected_lang)
                    else:
                        video_synopsis_text = active_llm.translate_text(f"Summarize this video:\n{all_text[:3000]}")
                    live_detail_box.caption("✅ Video synopsis generated.")
                except Exception as syn_err:
                    live_detail_box.caption(f"ℹ️ Video synopsis note: {syn_err}")

        overall_progress.progress(1.0)
        elapsed = time.time() - t0

        # Save to Disk in Dedicated Folder
        stem_name = Path(media_display_name).stem
        run_output_dir = OUTPUTS_DIR / stem_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        if pipeline_mode.startswith("🎬"):
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

            if video_synopsis_text:
                syn_path = run_output_dir / f"{stem_name}_video_synopsis.md"
                syn_path.write_text(video_synopsis_text, encoding="utf-8")
                saved_paths["video_synopsis"] = syn_path

            json_path = run_output_dir / f"{stem_name}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(final_result, jf, indent=2, ensure_ascii=False)
            saved_paths["json"] = json_path

        else:
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")

            notes_ar = generate_meeting_notes_markdown(
                final_result,
                title=f"Meeting Minutes - {stem_name}",
                use_translated=False,
                gemini_api_key=gemini_key_input,
                meeting_date=today,
                use_llm=use_llm_pipeline,
            )
            md_ar_path = run_output_dir / f"{stem_name}_meeting_notes_ar.md"
            md_ar_path.write_text(notes_ar, encoding="utf-8")
            saved_paths["notes_ar_md"] = md_ar_path

            saved_paths["_llm_used"] = use_llm_pipeline and ("## ✅ Decisions" in notes_ar or "## 🎯 Action Items" in notes_ar or "## Meeting Overview" in notes_ar)

            if enable_trans:
                notes_trans = generate_meeting_notes_markdown(
                    final_result,
                    title=f"Meeting Notes (EN) - {stem_name}",
                    use_translated=True,
                    target_lang=target_lang_code,
                    gemini_api_key=gemini_key_input,
                    meeting_date=today,
                    use_llm=use_llm_pipeline,
                )
                md_trans_path = run_output_dir / f"{stem_name}_meeting_notes_{target_lang_code}.md"
                md_trans_path.write_text(notes_trans, encoding="utf-8")
                saved_paths["notes_trans_md"] = md_trans_path
                main_notes = notes_trans
            else:
                main_notes = notes_ar

            txt_path = run_output_dir / f"{stem_name}_meeting_notes.txt"
            txt_path.write_text(main_notes, encoding="utf-8")
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
        tab_titles = ["📝 Subtitles Preview"]
        if "video_synopsis" in saved_paths:
            tab_titles.append("🎬 Video Synopsis")
        tab_titles.extend(["🔤 Arabic SRT", "🌍 English SRT", "📑 Bilingual SRT"])

        sub_tabs = st.tabs(tab_titles)
        
        with sub_tabs[0]:
            cues = generate_subtitles_from_segments(res.get("segments", []), 38, 2, use_translated=True, bilingual=("bilingual_srt" in saved_paths))
            st.caption(f"Displaying {len(cues)} tightly synchronized broadcast subtitle cues:")
            for c in cues[:25]:
                spk_tag = f"<b>[{c.get('speaker', 'Speaker')}]</b> " if c.get('speaker') else ""
                st.markdown(
                    f"""
                    <div class="sub-card">
                        <div class="sub-time">⏱️ {format_timestamp(c['start'])} ➔ {format_timestamp(c['end'])}</div>
                        <div class="sub-text">{spk_tag}{c['text'].replace(chr(10), '<br>')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if len(cues) > 25:
                st.caption(f"_...and {len(cues)-25} more subtitle cues saved in the output file._")

        cur_idx = 1
        if "video_synopsis" in saved_paths:
            with sub_tabs[cur_idx]:
                st.markdown(
                    '<span class="llm-badge llm-badge-local" style="margin-bottom: 0.8rem; display: inline-block;">'
                    '🧠 AI Video Intelligence</span>',
                    unsafe_allow_html=True
                )
                syn_content = saved_paths["video_synopsis"].read_text(encoding="utf-8")
                st.markdown(syn_content)
                st.download_button("⬇️ Download Synopsis (Markdown)", syn_content, file_name=saved_paths["video_synopsis"].name)
            cur_idx += 1

        with sub_tabs[cur_idx]:
            if "orig_srt" in saved_paths:
                st.text_area("Arabic SRT Content", saved_paths["orig_srt"].read_text(encoding="utf-8"), height=350)
                st.download_button("⬇️ Download Arabic SRT", saved_paths["orig_srt"].read_text(encoding="utf-8"), file_name=saved_paths["orig_srt"].name)
        cur_idx += 1

        with sub_tabs[cur_idx]:
            if "trans_srt" in saved_paths:
                st.text_area("English SRT Content", saved_paths["trans_srt"].read_text(encoding="utf-8"), height=350)
                st.download_button("⬇️ Download English SRT", saved_paths["trans_srt"].read_text(encoding="utf-8"), file_name=saved_paths["trans_srt"].name)
        cur_idx += 1

        with sub_tabs[cur_idx]:
            if "bilingual_srt" in saved_paths:
                st.text_area("Bilingual SRT Content", saved_paths["bilingual_srt"].read_text(encoding="utf-8"), height=350)
                st.download_button("⬇️ Download Bilingual SRT", saved_paths["bilingual_srt"].read_text(encoding="utf-8"), file_name=saved_paths["bilingual_srt"].name)

    else:
        # Meeting Notes View with Dedicated Tabs
        was_llm = saved_paths.get("_llm_used", False)
        tab_list = ["📋 Meeting Minutes"]
        if "notes_trans_md" in saved_paths:
            tab_list.append("🌍 Translated Notes")
        tab_list.append("📄 Plain Text")

        tabs = st.tabs(tab_list)

        with tabs[0]:
            if "notes_ar_md" in saved_paths:
                if was_llm:
                    st.markdown(
                        '<span class="llm-badge llm-badge-local" style="margin-bottom: 0.8rem; display: inline-block;">'
                        '🧠 Adaptive Executive Meeting Minutes</span>',
                        unsafe_allow_html=True
                    )
                notes_ar_text = saved_paths["notes_ar_md"].read_text(encoding="utf-8")
                st.markdown(notes_ar_text)
                st.download_button("⬇️ Download Meeting Minutes (Markdown)", notes_ar_text, file_name=saved_paths["notes_ar_md"].name)

        if "notes_trans_md" in saved_paths and len(tabs) > 1:
            with tabs[1]:
                notes_trans_text = saved_paths["notes_trans_md"].read_text(encoding="utf-8")
                st.markdown(notes_trans_text)
                st.download_button("⬇️ Download Translated Notes (Markdown)", notes_trans_text, file_name=saved_paths["notes_trans_md"].name)

        with tabs[-1]:
            if "notes_txt" in saved_paths:
                txt_content = saved_paths["notes_txt"].read_text(encoding="utf-8")
                st.text_area("Raw Meeting Minutes", txt_content, height=400)
                st.download_button("⬇️ Download Plain Text (.txt)", txt_content, file_name=saved_paths["notes_txt"].name)
