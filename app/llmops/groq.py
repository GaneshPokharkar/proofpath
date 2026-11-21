"""Groq: open models (Llama, GPT-OSS) on fast hardware, through an OpenAI-compatible API, with a
free tier. Implements the same two methods as GeminiLLM, so the rest of the app can't tell them
apart:

- extract(system, user, schema): structured output by forcing one function call whose
  parameters are the Pydantic schema, then validating it (one retry with the error).
- chat(system, contents, tools, final): one agent step. The agent speaks Gemini's message
  format, so this translates Gemini contents and function declarations to OpenAI messages and
  tools, and the reply back.
"""

import json
import time

import httpx
from pydantic import ValidationError

from app.llm import LLMError, T, _inline_refs

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _lower_types(node):
    """Gemini declarations use OpenAPI type names (OBJECT, STRING); JSON Schema wants lowercase."""
    if isinstance(node, dict):
        return {
            k: (v.lower() if k == "type" and isinstance(v, str) else _lower_types(v)) for k, v in node.items()
        }
    if isinstance(node, list):
        return [_lower_types(v) for v in node]
    return node


def to_openai_tools(declarations: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": _lower_types(d["parameters"]),
            },
        }
        for d in declarations
    ]


def to_openai_messages(system: str, contents: list[dict]) -> list[dict]:
    """Gemini contents -> OpenAI messages. Function calls get ids; the function responses that
    follow are matched to them in order, which is how Gemini pairs them too."""
    messages: list[dict] = [{"role": "system", "content": system}]
    pending: list[str] = []
    n = 0
    for c in contents:
        parts = c.get("parts", [])
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        results = [p["functionResponse"] for p in parts if "functionResponse" in p]
        texts = "".join(p.get("text", "") for p in parts if "text" in p)
        if c.get("role") == "model":
            msg: dict = {"role": "assistant", "content": texts or None}
            if calls:
                pending = []
                msg["tool_calls"] = []
                for fc in calls:
                    n += 1
                    cid = f"call_{n}"
                    pending.append(cid)
                    msg["tool_calls"].append(
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": fc["name"], "arguments": json.dumps(fc.get("args") or {})},
                        }
                    )
            messages.append(msg)
        elif results:
            for i, fr in enumerate(results):
                cid = pending[i] if i < len(pending) else f"call_orphan_{i}"
                messages.append(
                    {"role": "tool", "tool_call_id": cid, "content": json.dumps(fr.get("response", {}))}
                )
        else:
            messages.append({"role": "user", "content": texts})
    return messages


def from_openai_message(msg: dict) -> dict:
    parts: list[dict] = []
    if msg.get("content"):
        parts.append({"text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        parts.append({"functionCall": {"name": tc["function"]["name"], "args": args}})
    return {"role": "model", "parts": parts}


class GroqLLM:
    def __init__(self, api_key: str, model: str, client: httpx.Client | None = None, max_retries: int = 1):
        self.model_name = model
        self._key = api_key
        self._client = client or httpx.Client(timeout=60)
        self._max_retries = max_retries
        self.last_usage = (0, 0)

    def _post(self, body: dict) -> dict:
        body = {"model": self.model_name, "temperature": 0} | body
        for attempt in range(self._max_retries + 1):
            try:
                r = self._client.post(GROQ_URL, json=body, headers={"authorization": f"Bearer {self._key}"})
            except httpx.TimeoutException:
                if attempt < self._max_retries:
                    continue
                raise LLMError("Groq timed out") from None
            # A short wait for a per-minute limit; a daily limit is left to the router's fallback.
            if r.status_code in (429, 503) and attempt < self._max_retries:
                wait = float(r.headers.get("retry-after", "2") or 2)
                if wait <= 5:
                    time.sleep(wait)
                    continue
            if r.status_code != 200:
                raise LLMError(f"Groq {r.status_code}: {r.text[:500]}")
            data = r.json()
            u = data.get("usage") or {}
            self.last_usage = (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            return data
        raise AssertionError("unreachable")

    def extract(self, system: str, user: str, schema: type[T]) -> T:
        tool = {
            "type": "function",
            "function": {
                "name": "record",
                "description": f"Record the result as {schema.__name__}.",
                "parameters": _inline_refs(schema.model_json_schema()),
            },
        }
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            data = self._post(
                {
                    "messages": messages,
                    "tools": [tool],
                    "tool_choice": {"type": "function", "function": {"name": "record"}},
                }
            )
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                raise LLMError(f"Groq {self.model_name} returned no tool call")
            raw = calls[0]["function"].get("arguments") or "{}"
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"output failed validation twice: {exc}") from exc
                messages += [
                    {"role": "assistant", "content": None, "tool_calls": calls[:1]},
                    {
                        "role": "tool",
                        "tool_call_id": calls[0]["id"],
                        "content": f"Invalid: {exc}. Call record again with valid arguments.",
                    },
                ]
        raise AssertionError("unreachable")

    def chat(self, system: str, contents: list[dict], tools: list[dict], final: bool = False) -> dict:
        data = self._post(
            {
                "messages": to_openai_messages(system, contents),
                "tools": to_openai_tools(tools),
                "tool_choice": "none" if final else "auto",
            }
        )
        return from_openai_message(data["choices"][0]["message"])
