from unittest.mock import MagicMock, patch

import torch

import coherex.asr as asr


def test_cuda_model_loads_directly_to_gpu_without_cpu_roundtrip():
    processor = MagicMock()
    processor.tokenizer.get_vocab.return_value = {"a": 1}
    model = MagicMock()
    model.config.supported_languages = ["en"]
    vad = MagicMock()
    with (
        patch.object(asr.AutoProcessor, "from_pretrained", return_value=processor),
        patch.object(asr.CohereAsrForConditionalGeneration, "from_pretrained", return_value=model) as load,
    ):
        pipeline = asr.load_model(
            "trusted/model",
            device="cuda",
            device_index=0,
            compute_type="float16",
            language="en",
            vad_model=vad,
        )
    kwargs = load.call_args.kwargs
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["device_map"] == {"": "cuda:0"}
    model.to.assert_not_called()
    assert pipeline.model is model
