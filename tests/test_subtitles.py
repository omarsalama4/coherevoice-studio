from coherex.subtitles import (
    generate_subtitles_from_segments,
    split_text_into_cues,
    validate_subtitle_cues,
)


def test_short_segment_never_creates_zero_duration_cues():
    cues = split_text_into_cues("one two three four five six", 0.0, 0.5, 10, 1)
    assert cues
    assert all(cue["end"] > cue["start"] for cue in cues)
    assert validate_subtitle_cues(cues, 10, 1)["valid"]


def test_long_unspaced_token_respects_line_limit():
    cues = split_text_into_cues("x" * 35, 0, 3, 10, 2)
    assert all(len(line) <= 10 for cue in cues for line in cue["text"].splitlines())


def test_many_chunks_in_tiny_window_remain_positive_and_contiguous():
    cues = split_text_into_cues("x" * 1000, 0, 0.05, 10, 1)
    assert all(cue["end"] > cue["start"] for cue in cues)
    assert all(cues[i]["start"] == cues[i - 1]["end"] for i in range(1, len(cues)))


def test_bilingual_tracks_share_timing_and_do_not_overlap():
    segments = [{
        "start": 0.0,
        "end": 4.0,
        "text": "this is the source sentence with several words",
        "original_text": "this is the source sentence with several words",
        "translated_text": "هذه ترجمة عربية تحتوي على عدة كلمات للاختبار",
    }]
    cues = generate_subtitles_from_segments(segments, 18, 2, bilingual=True)
    assert all(len(cue["text"].splitlines()) <= 2 for cue in cues)
    assert validate_subtitle_cues(cues, 18, 2)["valid"]


def test_adjacent_overlapping_segments_are_serialized():
    segments = [
        {"start": 0, "end": 2, "text": "first"},
        {"start": 1, "end": 3, "text": "second"},
    ]
    cues = generate_subtitles_from_segments(segments)
    assert cues[1]["start"] >= cues[0]["end"]
