import json

import httpx
import pytest

from agent.llm_client import GeminiClient, LLMError, OpenRouterClient


class NoLimit:
    async def acquire(self):
        pass


def _gemini_ok(text="hello"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


async def test_gemini_uses_header_auth_and_retries_429():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "slow down",
                                                       "details": [{"@type": "x/RetryInfo", "retryDelay": "0s"}]}})
        return httpx.Response(200, json=_gemini_ok("done"))

    client = GeminiClient("secret", "model-x", NoLimit(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    resp = await client.generate("sys", [{"role": "user", "text": "hi"}])
    assert resp.text == "done" and len(seen) == 2
    assert seen[0].headers["x-goog-api-key"] == "secret"
    assert "key=" not in str(seen[0].url)
    body = json.loads(seen[0].content)
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


async def test_gemini_parses_function_calls():
    def handler(request):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "evolution", "args": {"generations": 2}}}]}}]})

    client = GeminiClient("k", "m", NoLimit(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    resp = await client.generate("", [{"role": "user", "text": "go"}], tools=[{"name": "evolution"}])
    assert resp.function_calls == [{"id": "call_0", "name": "evolution", "args": {"generations": 2}}]


async def test_non_json_error_and_402_are_clear():
    def handler(request):
        return httpx.Response(502, text="<html>bad gateway</html>")

    client = GeminiClient("k", "m", NoLimit(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(LLMError) as e:
        await client.generate("", [{"role": "user", "text": "x"}])
    assert "502" in str(e.value)

    def credits(request):
        return httpx.Response(402, json={"error": {"code": 402, "message": "needs credits"}})

    orc = OpenRouterClient("k", "org/model", NoLimit(), httpx.AsyncClient(transport=httpx.MockTransport(credits)))
    with pytest.raises(LLMError) as e:
        await orc.generate("", [{"role": "user", "text": "x"}])
    assert "credits" in str(e.value)
