# CohereX Audit Remediation

Date: 2026-09-05

## Resolved

- Fixed broken Gemini prompt imports, provider argument mismatches, and obsolete
  default model selection.
- Implemented the advertised automatic language-detection path in the UI and
  standalone CLIs.
- Fixed CLI translation so translated tracks are actually generated.
- Preserved source transcript text and separated translated/display fields.
- Removed silent translation-success fallbacks and unofficial translation
  dependencies/endpoints.
- Rebuilt subtitle segmentation with positive, contiguous timings, synchronized
  bilingual cues, enforced line limits, and JSON QC reports.
- Removed arbitrary-token wildcard alignment for unsupported/code-switched text.
- Fixed the missing diarization import, indexed overlap search, and fabricated
  nearest-speaker assignment default.
- Added evidence timestamp validation and prompt-injection boundaries for MOM.
- Made video synopsis claims accurately transcript-only.
- Bound the UI to localhost, restored CORS/XSRF protection, removed external font
  requests and personal default paths, randomized/cleaned temporary files, and
  made output directories collision-safe.
- Repaired package metadata, the Conda environment, launcher diagnostics,
  dependency constraints, tests, and CI.
- Removed known vulnerable NLTK/deep-translator dependencies and backported the
  unreleased Lightning checkpoint-instantiator allowlist.

## Validation performed

- `python scripts/doctor.py`: passed; `pip check` reports no broken requirements.
- `python -m pytest -q`: 14 passed.
- Streamlit AppTest: no exceptions.
- Live Streamlit health endpoint: `ok` on `127.0.0.1`.
- `pip-audit`: no known vulnerabilities after the two documented, mitigated
  exceptions in `SECURITY.md`.
- All Python files parse and `git diff --check` passes.

## Release gates requiring representative data

The code defects are remediated, but production-quality claims still require a
versioned evaluation corpus. Before a public or multi-user release, measure WER,
speaker DER, alignment error, translation review scores, MOM evidence precision,
latency, VRAM/RAM peaks, and subtitle CPS on representative Arabic, Egyptian,
English, noisy, overlapping, and code-switched recordings. The bundled server
remains intentionally local-only; network deployment additionally requires an
authenticated reverse proxy, TLS, per-user isolation, quotas, and a job queue.
