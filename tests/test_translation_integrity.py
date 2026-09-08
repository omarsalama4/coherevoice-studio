import pytest

import coherex.translator as translator


def test_unofficial_translation_is_opt_in(monkeypatch):
    monkeypatch.delenv("COHEREX_ALLOW_UNOFFICIAL_TRANSLATION", raising=False)
    with pytest.raises(translator.TranslationUnavailableError):
        translator.raw_neural_translate("مرحبا", "ar", "en")


def test_translation_preserves_source_text(monkeypatch):
    monkeypatch.setattr(
        translator,
        "llm_translate_batch",
        lambda **kwargs: ["Hello world"],
    )
    source = {"language": "ar", "segments": [{"start": 0, "end": 1, "text": "مرحبا بالعالم"}]}
    result = translator.translate_result(source, target_lang="en")
    segment = result["segments"][0]
    assert source["segments"][0]["text"] == "مرحبا بالعالم"
    assert segment["text"] == "مرحبا بالعالم"
    assert segment["translated_text"] == "Hello world"


def test_arabic_source_cannot_be_mislabeled_as_english(monkeypatch):
    monkeypatch.setattr(
        translator,
        "llm_translate_batch",
        lambda **kwargs: ["مرحبا بالعالم"],
    )
    source = {"language": "ar", "segments": [{"start": 0, "end": 1, "text": "مرحبا بالعالم"}]}
    with pytest.raises(translator.TranslationUnavailableError, match="copied the Arabic source"):
        translator.translate_result(source, target_lang="en")


def test_translation_count_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(translator, "llm_translate_batch", lambda **kwargs: ["Only one"])
    source = {
        "language": "ar",
        "segments": [
            {"start": 0, "end": 1, "text": "الأول"},
            {"start": 1, "end": 2, "text": "الثاني"},
        ],
    }
    with pytest.raises(translator.TranslationUnavailableError, match="count mismatch"):
        translator.translate_result(source, target_lang="en")
