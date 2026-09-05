import importlib


def _lazy_import(name):
    module = importlib.import_module(f"coherex.{name}")
    return module


def load_align_model(*args, **kwargs):
    alignment = _lazy_import("alignment")
    return alignment.load_align_model(*args, **kwargs)


def align(*args, **kwargs):
    alignment = _lazy_import("alignment")
    return alignment.align(*args, **kwargs)


def load_model(*args, **kwargs):
    asr = _lazy_import("asr")
    return asr.load_model(*args, **kwargs)


def load_audio(*args, **kwargs):
    audio = _lazy_import("audio")
    return audio.load_audio(*args, **kwargs)


def assign_word_speakers(*args, **kwargs):
    diarize = _lazy_import("diarize")
    return diarize.assign_word_speakers(*args, **kwargs)


def DiarizationPipeline(*args, **kwargs):
    diarize = _lazy_import("diarize")
    return diarize.DiarizationPipeline(*args, **kwargs)


def detect_language(*args, **kwargs):
    langid = _lazy_import("langid")
    return langid.detect_language(*args, **kwargs)


def extract_audio_from_video(*args, **kwargs):
    extractor = _lazy_import("extract_audio")
    return extractor.extract_audio_from_video(*args, **kwargs)


def translate_result(*args, **kwargs):
    trans = _lazy_import("translator")
    return trans.translate_result(*args, **kwargs)


def generate_subtitles_from_segments(*args, **kwargs):
    subs = _lazy_import("subtitles")
    return subs.generate_subtitles_from_segments(*args, **kwargs)


def generate_meeting_notes_markdown(*args, **kwargs):
    notes = _lazy_import("meeting_notes")
    return notes.generate_meeting_notes_markdown(*args, **kwargs)


def setup_logging(*args, **kwargs):
    logging_module = _lazy_import("log_utils")
    return logging_module.setup_logging(*args, **kwargs)


def get_logger(*args, **kwargs):
    logging_module = _lazy_import("log_utils")
    return logging_module.get_logger(*args, **kwargs)


def get_llm_client(*args, **kwargs):
    llm = _lazy_import("llm")
    return llm.get_llm_client(*args, **kwargs)


def is_llm_available(*args, **kwargs):
    llm = _lazy_import("llm")
    return llm.is_llm_available(*args, **kwargs)
