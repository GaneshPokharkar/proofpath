"""LLM access with schema-validated structured output.

Structured output works by forcing the model to call a single tool whose input schema is the
Pydantic model's JSON schema; the tool input is then validated by Pydantic. If validation fails,
the model gets the validation error and one more try, and a second failure raises.
"""

import json
import time
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    model_name: str

    def extract(self, system: str, user: str, schema: type[T]) -> T: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 2048):
        import anthropic  # imported lazily so tests and ingestion don't need the SDK configured

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
        self._max_tokens = max_tokens

    def extract(self, system: str, user: str, schema: type[T]) -> T:
        tool = {
            "name": "record",
            "description": f"Record the result as {schema.__name__}.",
            "input_schema": schema.model_json_schema(),
        }
        messages: list[dict] = [{"role": "user", "content": user}]
        for attempt in range(2):
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self._max_tokens,
                system=system,
                tools=[tool],
                tool_choice={"type": "tool", "name": "record"},
                messages=messages,
            )
            call = next((b for b in response.content if b.type == "tool_use"), None)
            if call is None:
                raise LLMError(f"{self.model_name} returned no tool call (stop: {response.stop_reason})")
            try:
                return schema.model_validate(call.input)
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"output failed validation twice: {exc}") from exc
                messages += [
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "is_error": True,
                                "content": f"Invalid input, fix and call record again:\n{exc}",
                            }
                        ],
                    },
                ]
        raise AssertionError("unreachable")


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _inline_refs(schema: dict) -> dict:
    """Gemini's response schema does not follow $ref/$defs, so expand them in place."""
    defs = schema.pop("$defs", {})

    def expand(node, in_properties: bool = False):
        if isinstance(node, dict):
            if "$ref" in node:
                return expand(dict(defs[node["$ref"].split("/")[-1]]))
            # "title" is a schema annotation to drop, except under "properties", where it is a
            # field name (e.g. a project's title) that must stay.
            return {
                k: expand(v, in_properties=(k == "properties"))
                for k, v in node.items()
                if in_properties or k != "title"
            }
        if isinstance(node, list):
            return [expand(v) for v in node]
        return node

    return expand(schema)


class GeminiLLM:
    """Google Gemini over REST with JSON-schema constrained output. Has a free tier, whose
    prompts Google may use to improve its products, so it suits public data like job postings."""

    def __init__(self, api_key: str, model: str, client: httpx.Client | None = None, max_retries: int = 4):
        self.model_name = model
        self._key = api_key
        self._client = client or httpx.Client(timeout=60)
        self._max_retries = max_retries

    def _post(self, body: dict) -> dict:
        for attempt in range(self._max_retries + 1):
            r = self._client.post(
                GEMINI_URL.format(model=self.model_name), json=body, headers={"x-goog-api-key": self._key}
            )
            # Free-tier rate limits (429) and transient overloads (503) are retried with backoff.
            if r.status_code in (429, 503) and attempt < self._max_retries:
                time.sleep(min(60, 2 ** (attempt + 2)))
                continue
            if r.status_code != 200:
                raise LLMError(f"Gemini {r.status_code}: {r.text[:500]}")
            return r.json()
        raise AssertionError("unreachable")

    def extract(self, system: str, user: str, schema: type[T]) -> T:
        contents = [{"role": "user", "parts": [{"text": user}]}]
        for attempt in range(2):
            data = self._post(
                {
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": contents,
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": _inline_refs(schema.model_json_schema()),
                        "temperature": 0,
                    },
                }
            )
            candidates = data.get("candidates") or []
            if not candidates or "content" not in candidates[0]:
                raise LLMError(f"Gemini returned no content: {json.dumps(data)[:500]}")
            raw = "".join(p.get("text", "") for p in candidates[0]["content"]["parts"])
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"output failed validation twice: {exc}") from exc
                contents += [
                    {"role": "model", "parts": [{"text": raw}]},
                    {
                        "role": "user",
                        "parts": [{"text": f"That JSON was invalid, return a fixed version:\n{exc}"}],
                    },
                ]
        raise AssertionError("unreachable")


def build_llm(provider: str, api_key: str | None, model: str) -> StructuredLLM:
    if not api_key:
        raise LLMError(f"no API key configured for LLM provider {provider!r}")
    if provider == "gemini":
        return GeminiLLM(api_key, model)
    if provider == "anthropic":
        return AnthropicLLM(api_key, model)
    raise LLMError(f"unknown LLM provider {provider!r} (expected gemini or anthropic)")
