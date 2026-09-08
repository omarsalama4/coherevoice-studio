import pytest

from coherex.meeting_notes import (
    _validate_markdown_evidence,
    _validate_structured_evidence,
    format_short_time,
    group_speaker_dialogue,
)


class Item:
    def __init__(self, timestamp):
        self.timestamp = timestamp


class Mom:
    decisions = [Item("00:00:05")]
    action_items = [Item("[00:00:10]")]


def test_timestamp_is_always_canonical():
    assert format_short_time(65) == "00:01:05"


def test_structured_evidence_rejects_invented_timestamp():
    with pytest.raises(ValueError, match="Ungrounded"):
        _validate_structured_evidence(Mom(), {"00:00:05"})


def test_translated_dialogue_never_silently_returns_source():
    with pytest.raises(RuntimeError, match="translated_text"):
        group_speaker_dialogue([{"start": 0, "end": 1, "text": "مرحبا"}], use_translated=True)


def test_markdown_evidence_rejects_uncited_decisions_and_actions():
    markdown = """# Minutes of Meeting
## Executive Summary
Summary.
## Problem
Problem details.
## Proposed Approach
Approach details.
## Decisions Made
- Hire a senior engineer.
## Action Items & Next Steps
- Call a consultant.
## Open Questions & Unresolved Issues
- None identified.
"""
    with pytest.raises(ValueError, match="Decisions.*without evidence"):
        _validate_markdown_evidence(markdown, {"00:00:05"})


def test_markdown_evidence_accepts_grounded_decisions_and_actions():
    markdown = """# Minutes of Meeting
## Executive Summary
Summary.
## Problem
Problem details.
## Proposed Approach
Approach details.
## Decisions Made
- Keep the prototype narrow. [00:00:05]
## Action Items & Next Steps
- Validate the concept. [00:00:10]
## Open Questions & Unresolved Issues
- Is the idea differentiated?
"""
    _validate_markdown_evidence(markdown, {"00:00:05", "00:00:10"})


def test_markdown_evidence_accepts_explicit_no_final_decisions():
    markdown = """# Minutes of Meeting
## Meeting Overview
Overview.
## Executive Summary
Summary.
## Problem
Problem details.
## Proposed Approach
Approach details.
## Decisions Made
No final decisions were identified.
## Action Items & Next Steps
No formal action items were identified.
## Open Questions & Unresolved Issues
- Is the idea differentiated?
"""
    _validate_markdown_evidence(markdown, {"00:00:05"})
