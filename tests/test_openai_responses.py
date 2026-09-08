import json

import coherex.llm as llm_router
from coherex.llm.openai_client import OpenAIClient
from coherex.llm.groq_client import GroqClient


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_official_openai_uses_responses_api(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"output_text": "API result"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIClient(api_key="test-secret", model="gpt-5.6-terra")
    result = client._chat_completion([
        {"role": "system", "content": "Be accurate."},
        {"role": "user", "content": "Summarize this."},
    ])

    assert result == "API result"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["payload"]["instructions"] == "Be accurate."
    assert captured["payload"]["input"] == [{"role": "user", "content": "Summarize this."}]
    assert captured["payload"]["store"] is False
    assert "temperature" not in captured["payload"]
    assert client.provider_name == "openai"


def test_compatible_endpoint_keeps_chat_completions(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "Compatible result"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIClient(
        api_key="test-secret",
        model="compatible-model",
        base_url="https://example.invalid/v1",
    )
    result = client._chat_completion([{"role": "user", "content": "Hello"}], json_mode=True)

    assert result == "Compatible result"
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert client.provider_name == "openai-compatible"


def test_translation_accepts_fenced_json(monkeypatch):
    client = OpenAIClient(api_key="test-secret")
    monkeypatch.setattr(
        client,
        "_chat_completion",
        lambda *args, **kwargs: '```json\n{"translations":["Hello"]}\n```',
    )
    assert client.translate_batch(["مرحبا"], source_lang="ar", target_lang="en") == ["Hello"]


def test_router_uses_explicit_groq_provider_only(monkeypatch):
    class _AvailableClient:
        def is_available(self):
            return True

    expected = _AvailableClient()
    monkeypatch.setattr(
        llm_router,
        "get_openai_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenAI queried")),
    )
    monkeypatch.setattr(llm_router, "get_groq_client", lambda *args, **kwargs: expected)

    assert llm_router.get_llm_client(provider="groq", groq_api_key="groq-secret") is expected


def test_groq_client_uses_hosted_groq_endpoint():
    client = GroqClient(api_key="groq-secret", model="openai/gpt-oss-120b")
    assert client.base_url == "https://api.groq.com/openai/v1"
    assert client.model == "openai/gpt-oss-120b"
    assert client.provider_name == "groq"
