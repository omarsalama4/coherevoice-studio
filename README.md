# 🎙️ CohereVoice Studio

**CohereVoice Studio** is an advanced AI-powered speech recognition, broadcast subtitle generation, and meeting intelligence suite. Built on top of Cohere's state-of-the-art transformer speech models, it delivers broadcast-grade subtitles, structured meeting minutes, speaker diarization, and dynamic multilingual translation with specialized optimization for **Egyptian Arabic & Arabic dialects**.

---

## 🌟 Key Features

- 🎬 **Dual Output Modes**:
  - **Broadcast Subtitles Mode**: Generates non-overlapping cues with enforced line limits and a machine-readable quality report.
  - **Meeting Notes & Summaries Mode**: Produces an adaptive professional MOM with an executive summary, topic-specific deep dives, grounded decisions, action items, and open questions.
- 🔑 **API-Backed Intelligence**:
  - Professional MOM, subtitle translation, and video synopsis support OpenAI, Groq, and Google Gemini.
  - The provider and model are selected explicitly in the UI or with `LLM_PROVIDER` and the provider-specific model variable.
  - Intelligence features fail closed when the selected provider has no API key; CohereX never silently spends against a different provider.
- 🧠 **Intelligent Automatic Model Routing**:
  - **Arabic Media (`ar`)**: Automatically routes to `CohereLabs/cohere-transcribe-arabic-07-2026` (finetuned for Egyptian/Arab dialects, colloquial slang, and rapid dialogue).
  - **Multilingual Media (`en`, `es`, `fr`, etc.)**: Automatically routes to `CohereLabs/cohere-transcribe-03-2026` (14-language base model).
- 🌍 **Dynamic Contextual Translation**:
  - Uses the selected OpenAI, Groq, or Gemini model; translation failures are explicit and never mislabeled as success.
  - Supports dual-track **Bilingual Subtitles** (Translated + Original).
- 👥 **Speaker Diarization & Word Alignment**:
  - Powered by PyAnnote and wav2vec 2.0 forced alignment; unalignable code-switched characters are safely interpolated.
- 💻 **Dual Interface**:
  - Elegant **Streamlit Web Studio** with live preview and auto-saving.
  - Scriptable **Command-Line Interface (CLI)** for automated workflows.

---

## 📂 Clean Project Structure

```
coherevoice-studio/
├── .env.example                    # Safe provider/API configuration template
├── app.py                          # Streamlit Studio Web UI
├── run_ui.bat                      # Windows 1-Click Launcher for Web UI
├── pyproject.toml                  # Python package configuration
├── README.md                       # User Guide & Documentation
├── TECHNICAL_ARCHITECTURE.md        # Deep-Dive Technical Reference & Pipeline Docs
├── environment.yml                # Reproducible Conda environment
├── coherex/                        # Core Python Engine Library
│   ├── __init__.py                 # Lazy export API
│   ├── asr.py                      # Cohere ASR Model Loader & Transcriber
│   ├── alignment.py                # wav2vec Phoneme Word Alignment
│   ├── diarize.py                  # PyAnnote Speaker Diarization
│   ├── extract_audio.py            # FFmpeg Audio Extractor & Normalizer
│   ├── subtitles.py                # Broadcast-Quality Subtitle Splitter (SRT/VTT)
│   ├── meeting_notes.py            # Meeting Notes & Dialogue Summarizer
│   ├── translator.py               # Dynamic Neural Subtitle & Transcript Translator
│   ├── llm/                         # OpenAI, Groq, and Gemini provider clients
│   └── utils.py                    # Timecode formatters & helpers
├── scripts/                        # Standalone CLI tools & Batch Runners
│   ├── cli_transcribe.py           # Command-Line Subtitle Generator
│   └── cli_meeting_notes.py        # Command-Line Meeting Notes Generator
└── outputs/                        # Default Auto-Save Directory for Generated Outputs
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (Conda recommended)
- NVIDIA GPU with CUDA support (or CPU mode)
- [FFmpeg](https://ffmpeg.org/) installed and available on system PATH
- An OpenAI, Groq, or Gemini API key for translation, video synopsis, and professional MOM generation

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/omarsalama4/coherevoice-studio.git
cd coherevoice-studio

# Create the tested environment, including FFmpeg
conda env create -f environment.yml
conda activate coherex

# Verify dependencies and that this checkout is the active package
python scripts/doctor.py
```

### 3. Authentication
Copy `.env.example` to `.env`, select one provider, and fill its API key. The real `.env` is ignored by Git.
```env
HF_TOKEN=hf_your_huggingface_token_here
LLM_PROVIDER=openai

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6-terra

# Alternative hosted providers:
# GROQ_API_KEY=gsk_...
# GROQ_MODEL=openai/gpt-oss-120b
# GEMINI_API_KEY=...
# GEMINI_MODEL=gemini-3.8-flash
```

---

## 🚀 How to Use

### Option A: Interactive Web UI (Streamlit Studio)

Launch the web studio with one click:
```bash
# Windows
run_ui.bat

# Or from terminal
streamlit run app.py --server.address 127.0.0.1
```

#### In the Web UI:
1. **Upload Media**: Drag-and-drop any video/movie (`.mp4`, `.mkv`, `.avi`, `.mov`) or audio file (`.mp3`, `.wav`, `.m4a`), or paste a local file path.
2. **Select Mode**:
   - **🎬 Subtitles & Captions**: Generates short, readable subtitle cues with custom line lengths.
   - **📋 Meeting Notes**: Generates an Executive Overview, Speaker Turns, and Action Items.
3. **Choose Intelligence Provider and Model**: Select OpenAI, Groq, or Gemini and enter that provider's API key.
4. **Toggle Translation (Optional)**: Choose target language (e.g. English, French, Spanish) and enable Bilingual Mode if desired.
5. **Click "🚀 Run Pipeline"**: Outputs are automatically saved to `outputs/` and available for download.

---

### Option B: Command-Line Interface (CLI)

#### 1. Generate Subtitles from Video or Audio
```bash
# Transcribe Arabic video to English + Arabic bilingual subtitles
python scripts/cli_transcribe.py "C:\path\to\movie.mp4" --lang ar --translate en --bilingual

# Transcribe English audio to Spanish subtitles
python scripts/cli_transcribe.py "C:\path\to\podcast.mp3" --lang en --translate es
```

#### 2. Generate Meeting Notes & Summaries
```bash
# Generate Meeting Notes from a recorded discussion
python scripts/cli_meeting_notes.py "C:\path\to\meeting.mp4" --lang ar --translate en
```

---

## 📄 Output Files Explained

Each run is saved under a unique timestamped directory in `outputs/`, preventing concurrent jobs from overwriting one another.

| File | Type | Description |
| :--- | :--- | :--- |
| `video_ar.srt` | Subtitles | Original language subtitles with short, comfortable cues (1–2 lines max). |
| `video_en.srt` | Subtitles | English translated subtitles dynamically localized and time-aligned. |
| `video_bilingual.srt` | Subtitles | Dual-track subtitles (English line on top, Arabic line below). |
| `video_ar.vtt` / `_en.vtt` | WebVTT | Web-standard subtitle format for HTML5 video players. |
| `video_meeting_notes_ar.md` | Markdown | Structured Meeting Minutes with Executive Summary, Highlights, and Speaker Turns. |
| `video_meeting_notes_en.md` | Markdown | English translated Meeting Notes for global teams. |
| `video_meeting_notes.txt` | Text | Plain-text dialogue minutes with timestamps and speaker labels. |
| `video.json` | Metadata | Full machine metadata (segment timestamps, confidence scores, speaker tags). |
| `video_subtitle_qc.json` | Quality report | Structural subtitle errors and reading-speed warnings. |
| `video_models_used.json` | Provenance | Exact ASR, alignment, diarization, API provider, and LLM model used for the run. |

## Security scope

The bundled UI is a local workstation tool bound to `127.0.0.1`. Do not expose it
to a network without authentication, TLS, isolation, and request limits. Uploaded
media is removed from temporary storage after each job; generated outputs remain
until you delete them. See [SECURITY.md](SECURITY.md).

---

## 📚 Technical Documentation

For an in-depth explanation of how Voice Activity Detection, ASR decoding, forced phoneme alignment, subtitle splitting heuristics, and dynamic neural translation work under the hood, read:
👉 **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)**

---

## 🙏 Credits & Acknowledgements

- **Original Foundation**: [bakrianoo/cohereX](https://github.com/bakrianoo/cohereX) by **[bakrianoo](https://github.com/bakrianoo)**.
- **ASR Models**: [CohereLabs Transcribe](https://huggingface.co/CohereLabs).
- **Speaker Diarization**: [PyAnnote Audio](https://github.com/pyannote/pyannote-audio).
- **Phoneme Alignment**: [WhisperX](https://github.com/m-bain/whisperX) & [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h).
