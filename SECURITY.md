# Security and privacy

CohereX is configured as a local workstation application. The bundled launcher
binds Streamlit to `127.0.0.1`; do not expose it to an untrusted network without
adding authentication, TLS, request limits, isolation, and a job queue.

Uploaded media and normalized audio are written to randomized per-job temporary
directories and removed after processing. Generated transcripts, subtitles, and
meeting notes remain in `outputs/` until the operator deletes them.

ASR remains on the workstation, but transcript text is sent to the selected OpenAI, Groq, or Gemini
API for translation, video synopsis, and professional meeting intelligence.
Review the selected provider's retention and privacy controls before processing
sensitive recordings. Unofficial public translation endpoints are not used.

Custom diarization model identifiers are rejected by default because ML model
artifacts can contain executable serialization formats. Review the artifacts
before opting in with `COHEREX_ALLOW_CUSTOM_MODELS=true`.

Never commit `.env` or `.streamlit/secrets.toml`. To report a vulnerability,
contact the repository owner privately and include reproduction steps without
real customer recordings, API keys, or other sensitive data.

## Temporary vulnerability exceptions

- `PYSEC-2026-3624` / CVE-2026-58659: `pyannote-audio` currently requires
  Lightning 2.6.5 and no fixed PyPI release exists. CohereX backports upstream
  commit `d710d68` at runtime in `coherex/security.py`, restricting checkpoint
  `_instantiator` values to Lightning's two trusted CLI factories. A regression
  test verifies that an `os.system` checkpoint is blocked.
- `PYSEC-2026-3447` / CVE-2026-59890: PyTorch currently constrains runtime
  setuptools below 82, while the fix is 83. The issue affects Unicode exclusion
  rules when building source distributions on macOS. CohereX uses setuptools 83+
  in its isolated build environment, CI builds on Linux, and the Windows runtime
  does not build source distributions. Remove this exception when PyTorch lifts
  its constraint.
