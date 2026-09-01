# 🎙️ CohereVoice Studio

**CohereVoice Studio** is an advanced AI-powered speech recognition, broadcast subtitle generation, and meeting intelligence suite. Built on top of Cohere's state-of-the-art transformer speech models, it delivers broadcast-grade subtitles, structured meeting minutes, speaker diarization, and dynamic multilingual translation with specialized optimization for **Egyptian Arabic & Arabic dialects**.

> [!NOTE]
> **Attribution & Acknowledgement**: This project builds upon and extends the foundational speech alignment and diarization architecture of [bakrianoo/cohereX](https://github.com/bakrianoo/cohereX) by **[bakrianoo](https://github.com/bakrianoo)**.

---

## 🌟 Key Features

- 🎬 **Dual Output Modes**:
  - **Broadcast Subtitles Mode**: Generates tightly synchronized, broadcast-standard subtitle cues (1–2 lines, max 38 chars/line, 1.5–5s durations) with zero screen-covering text walls.
  - **Meeting Notes & Summaries Mode**: Produces structured Executive Overviews, Key Highlights, Speaker-by-Speaker Discussion Minutes, and Topic Timelines.
- 🧠 **Intelligent Automatic Model Routing**:
  - **Arabic Media (`ar`)**: Automatically routes to `CohereLabs/cohere-transcribe-arabic-07-2026` (finetuned for Egyptian/Arab dialects, colloquial slang, and rapid dialogue).
  - **Multilingual Media (`en`, `es`, `fr`, etc.)**: Automatically routes to `CohereLabs/cohere-transcribe-03-2026` (14-language base model).
- 🌍 **Dynamic Contextual Translation**:
  - Fast, general-purpose neural translation that translates full meaning dynamically rather than literal word-by-word.
  - Supports dual-track **Bilingual Subtitles** (Translated + Original).
- 👥 **Speaker Diarization & Word Alignment**:
  - Powered by PyAnnote and wav2vec 2.0 phoneme alignment for millisecond-exact word timing and speaker labels.
- 💻 **Dual Interface**:
  - Elegant **Streamlit Web Studio** with live preview and auto-saving.
  - Scriptable **Command-Line Interface (CLI)** for automated workflows.

---

## 📂 Clean Project Structure

```
coherevoice-studio/
├── app.py                          # Streamlit Studio Web UI
├── run_ui.bat                      # Windows 1-Click Launcher for Web UI
├── pyproject.toml                  # Python package configuration
├── README.md                       # User Guide & Documentation
├── TECHNICAL_ARCHITECTURE.md        # Deep-Dive Technical Reference & Pipeline Docs
├── .env                            # Environment keys (HF_TOKEN)
├── coherex/                        # Core Python Engine Library
│   ├── __init__.py                 # Lazy export API
│   ├── asr.py                      # Cohere ASR Model Loader & Transcriber
│   ├── alignment.py                # wav2vec Phoneme Word Alignment
│   ├── diarize.py                  # PyAnnote Speaker Diarization
│   ├── extract_audio.py            # FFmpeg Audio Extractor & Normalizer
│   ├── subtitles.py                # Broadcast-Quality Subtitle Splitter (SRT/VTT)
│   ├── meeting_notes.py            # Meeting Notes & Dialogue Summarizer
│   ├── translator.py               # Dynamic Neural Subtitle & Transcript Translator
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

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/omarsalama4/coherevoice-studio.git
cd coherevoice-studio

# Create and activate environment
conda create -n coherevoice python=3.11 -y
conda activate coherevoice

# Install dependencies and local package
pip install -e .
```

### 3. Authentication
Create a `.env` file in the root folder with your Hugging Face API key:
```env
HF_TOKEN=hf_your_huggingface_token_here
```

---

## 🚀 How to Use

### Option A: Interactive Web UI (Streamlit Studio)

Launch the web studio with one click:
```bash
# Windows
run_ui.bat

# Or from terminal
streamlit run app.py
```

#### In the Web UI:
1. **Upload Media**: Drag-and-drop any video/movie (`.mp4`, `.mkv`, `.avi`, `.mov`) or audio file (`.mp3`, `.wav`, `.m4a`), or paste a local file path.
2. **Select Mode**:
   - **🎬 Subtitles & Captions**: Generates short, readable subtitle cues with custom line lengths.
   - **📋 Meeting Notes**: Generates an Executive Overview, Speaker Turns, and Action Items.
3. **Toggle Translation (Optional)**: Choose target language (e.g. English, French, Spanish) and enable Bilingual Mode if desired.
4. **Click "🚀 Run Pipeline"**: Outputs are automatically saved to `outputs/` and available for download.

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

When processing any file (e.g. `video.mp4`), all output tracks are auto-saved directly to `outputs/`:

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
