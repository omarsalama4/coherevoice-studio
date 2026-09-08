import pytest

from coherex.security import harden_lightning_checkpoint_loading
from coherex.diarize import DiarizationPipeline


def test_lightning_untrusted_instantiator_is_blocked():
    harden_lightning_checkpoint_loading()
    import lightning.pytorch as pl
    import lightning.pytorch.core.saving as saving

    class Example(pl.LightningModule):
        def __init__(self):
            super().__init__()

    checkpoint = {
        "state_dict": {},
        "hyper_parameters": {"_instantiator": "os.system"},
    }
    with pytest.raises(ValueError, match="Blocked untrusted"):
        saving._load_state(Example, checkpoint, strict=False)


def test_custom_diarization_model_requires_opt_in(monkeypatch):
    monkeypatch.delenv("COHEREX_ALLOW_CUSTOM_MODELS", raising=False)
    with pytest.raises(ValueError, match="Untrusted diarization model"):
        DiarizationPipeline(model_name="unknown-owner/unreviewed-checkpoint")
