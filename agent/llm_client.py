from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass
class LLMResponse:
    text: str = ""
    function_calls: list = field(default_factory=list)  # [{"id": str, "name": str, "args": dict}]
    raw: list | None = None  # provider-specific passthrough (Gemini parts)


class LLMClient:
    provider_name: str = "base"

    async def generate(
        self, system_instruction: str, history: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


class LLMError(RuntimeError):
    """A request that failed for good: bad key, no credits, malformed reply."""


def create_client(provider: str, api_key: str, model: str, rate_limiter) -> LLMClient:
    provider = provider.strip().lower()
    if provider == "gemini":
        return GeminiClient(api_key=api_key, model=model, rate_limiter=rate_limiter)
    if provider == "openrouter":
        return OpenRouterClient(api_key=api_key, model=model, rate_limiter=rate_limiter)
    raise ValueError(f"unknown LLM provider '{provider}' (expected 'gemini' or 'openrouter')")


_ROLE_DEFAULTS = {
    "gemini": {"chat": "gemini-flash-lite-latest", "evolution": "gemini-flash-latest"},
    "openrouter": {
        "chat": "meta-llama/llama-3.3-70b-instruct",
        "evolution": "meta-llama/llama-3.3-70b-instruct",
    },
}


def api_key_env_var_for(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == "gemini":
        return "GEMINI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    raise ValueError(f"unknown LLM provider '{provider}'")


@dataclass
class ModelRoles:
    """Two roles: `chat` runs the agent loop (high volume), `evolution` writes code."""

    chat_provider: str
    chat_model: str
    chat_api_key: str
    evolution_provider: str
    evolution_model: str
    evolution_api_key: str

    def same_endpoint(self) -> bool:
        return (
            self.chat_provider == self.evolution_provider
            and self.chat_model == self.evolution_model
            and self.chat_api_key == self.evolution_api_key
        )

    def describe(self) -> str:
        if self.same_endpoint():
            return f"chat + evolution: {self.chat_provider} ({self.chat_model})"
        return (
            f"chat: {self.chat_provider} ({self.chat_model})  |  "
            f"evolution: {self.evolution_provider} ({self.evolution_model})"
        )

    @classmethod
    def from_env_and_choice(cls, chat_provider: str, evolution_provider: str) -> ModelRoles:
        chat_provider = chat_provider.strip().lower()
        evolution_provider = evolution_provider.strip().lower()
        return cls(
            chat_provider=chat_provider,
            chat_model=os.environ.get("CHAT_MODEL") or _ROLE_DEFAULTS[chat_provider]["chat"],
            chat_api_key=os.environ.get(api_key_env_var_for(chat_provider), ""),
            evolution_provider=evolution_provider,
            evolution_model=os.environ.get("EVOLUTION_MODEL") or _ROLE_DEFAULTS[evolution_provider]["evolution"],
            evolution_api_key=os.environ.get(api_key_env_var_for(evolution_provider), ""),
        )

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.chat_api_key:
            missing.append(api_key_env_var_for(self.chat_provider))
        if not self.evolution_api_key:
            name = api_key_env_var_for(self.evolution_provider)
            if name not in missing:
                missing.append(name)
        return missing


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


async def _post_with_retries(client: httpx.AsyncClient, rate_limiter, provider: str, **request) -> dict:
    """POSTs under the rate limiter and retries transient failures. Returns parsed JSON."""
    attempt = 0
    while True:
        await rate_limiter.acquire()
        try:
            resp = await client.post(**request)
        except httpx.HTTPError as e:
            if attempt >= _MAX_RETRIES:
                raise LLMError(f"{provider} request failed after {attempt} retries: {type(e).__name__}: {e or 'no detail'}") from e
            attempt += 1
            delay = 2.0 * attempt
            log.warning("[%s] transport error (%s: %s); retrying in %.0fs (%d/%d)",
                        provider, type(e).__name__, e or "no detail", delay, attempt, _MAX_RETRIES)
            await asyncio.sleep(delay)
            continue

        try:
            data = resp.json()
        except ValueError:
            data = {"error": {"message": _short(resp.text), "code": resp.status_code}}

        if resp.status_code < 400 and "error" not in data:
            return data

        error = data.get("error") or {}
        message = error.get("message") if isinstance(error, dict) else str(error)
        status = error.get("status", "") if isinstance(error, dict) else ""
        retryable = resp.status_code in _RETRYABLE_STATUS or status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE")
        if retryable and attempt < _MAX_RETRIES:
            attempt += 1
            delay = _retry_delay(resp, data, attempt)
            log.warning("[%s] HTTP %s (%s); retrying in %.1fs (%d/%d)",
                        provider, resp.status_code, _short(message or "", 120), delay, attempt, _MAX_RETRIES)
            await asyncio.sleep(delay)
            continue

        if resp.status_code == 402:
            raise LLMError(f"{provider} has no credits left for this request: {_short(message or '')}")
        raise LLMError(f"{provider} request failed [{resp.status_code}]: {_short(message or resp.text)}")


def _retry_delay(resp: httpx.Response, data: dict, attempt: int) -> float:
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    for detail in (data.get("error") or {}).get("details", []) or []:
        if isinstance(detail, dict) and detail.get("@type", "").endswith("RetryInfo"):
            try:
                return float(str(detail.get("retryDelay", "10s")).rstrip("s")) + 0.5
            except ValueError:
                pass
    return min(60.0, 5.0 * attempt)


class GeminiClient(LLMClient):
    provider_name = "gemini"
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str, rate_limiter, http_client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.rate_limiter = rate_limiter
        self._owns_client = http_client is None
        self.client = http_client or httpx.AsyncClient(timeout=120.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _to_contents(history: list[dict]) -> list[dict]:
        contents = []
        for turn in history:
            role = turn["role"]
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": turn["text"]}]})
            elif role == "assistant":
                raw_parts = turn.get("raw")
                if raw_parts is not None:
                    contents.append({"role": "model", "parts": raw_parts})
                else:
                    parts = [{"text": turn["text"]}] if turn.get("text") else []
                    parts += [{"functionCall": {"name": fc["name"], "args": fc["args"]}}
                              for fc in turn.get("function_calls", [])]
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": turn["name"], "response": {"result": turn["result"]}}}],
                })
            else:
                raise ValueError(f"unknown history role: {role}")
        return contents

    async def generate(
        self, system_instruction: str, history: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        body: dict = {"contents": self._to_contents(history)}
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]

        data = await _post_with_retries(
            self.client, self.rate_limiter, self.provider_name,
            url=f"{self.API_URL}/{self.model}:generateContent",
            json=body,
            headers={"x-goog-api-key": self.api_key},
        )
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"gemini returned no candidates{f' (blocked: {reason})' if reason else ''}")
        return self._parse_candidate(candidates[0])

    @staticmethod
    def _parse_candidate(candidate: dict) -> LLMResponse:
        parts = candidate.get("content", {}).get("parts", [])
        text = ""
        function_calls = []
        for i, part in enumerate(parts):
            if "functionCall" in part:
                fc = part["functionCall"]
                function_calls.append({"id": f"call_{i}", "name": fc.get("name", ""), "args": fc.get("args", {})})
            elif "text" in part:
                text += part["text"]
        return LLMResponse(text=text, function_calls=function_calls, raw=parts)


class OpenRouterClient(LLMClient):
    provider_name = "openrouter"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str, rate_limiter, http_client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.rate_limiter = rate_limiter
        self._owns_client = http_client is None
        self.client = http_client or httpx.AsyncClient(timeout=120.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _to_messages(system_instruction: str, history: list[dict]) -> list[dict]:
        messages: list[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        for turn in history:
            role = turn["role"]
            if role == "user":
                messages.append({"role": "user", "content": turn["text"]})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": turn.get("text") or None}
                function_calls = turn.get("function_calls", [])
                if function_calls:
                    msg["tool_calls"] = [
                        {
                            "id": fc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {"name": fc["name"], "arguments": json.dumps(fc["args"])},
                        }
                        for i, fc in enumerate(function_calls)
                    ]
                messages.append(msg)
            elif role == "tool":
                messages.append({"role": "tool", "tool_call_id": turn.get("id", "call_0"), "content": turn["result"]})
            else:
                raise ValueError(f"unknown history role: {role}")
        return messages

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    async def generate(
        self, system_instruction: str, history: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        body: dict = {"model": self.model, "messages": self._to_messages(system_instruction, history)}
        if tools:
            body["tools"] = self._to_openai_tools(tools)
            body["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://github.com/"),
            "X-Title": os.environ.get("OPENROUTER_APP_NAME", "CompeteEvolve"),
        }
        data = await _post_with_retries(
            self.client, self.rate_limiter, self.provider_name, url=self.API_URL, json=body, headers=headers
        )
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("openrouter returned no choices")
        return self._parse_choice(choices[0])

    @staticmethod
    def _parse_choice(choice: dict) -> LLMResponse:
        message = choice.get("message", {})
        text = message.get("content") or ""
        function_calls = []
        for tc in message.get("tool_calls") or []:
            function = tc.get("function", {})
            raw_args = function.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
            function_calls.append({"id": tc.get("id", ""), "name": function.get("name", ""), "args": args})
        return LLMResponse(text=text, function_calls=function_calls)
