import json

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy import text

from app.llm import GeminiChat, LLMError
from app.llmops import trace
from app.llmops.groq import GroqLLM, from_openai_message, to_openai_messages, to_openai_tools
from app.llmops.router import Route, RoutedLLM, build_router, retryable


class Out(BaseModel):
    answer: str


class ProjectIdeas(BaseModel):  # a "heavy" schema name
    answer: str


@pytest.fixture
def records(monkeypatch):
    got = []
    monkeypatch.setattr(trace, "sink", got.append)
    return got


def groq_client(handler):
    return GroqLLM(
        "k", "llama-test", client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0
    )


def gemini_client(handler):
    return GeminiChat(
        "k", "gemini-test", client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0
    )


def groq_tool_reply(args: dict, usage=(11, 7)):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "record", "arguments": json.dumps(args)},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        },
    )


def gemini_json_reply(obj: dict):
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(obj)}]}}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5},
        },
    )


def test_groq_structured_output_and_usage():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return groq_tool_reply({"answer": "ok"})

    c = groq_client(handler)
    assert c.extract("sys", "user", Out).answer == "ok"
    assert seen["body"]["tool_choice"] == {"type": "function", "function": {"name": "record"}}
    assert c.last_usage == (11, 7)


def test_groq_retries_once_on_invalid_output():
    replies = iter([groq_tool_reply({"wrong": 1}), groq_tool_reply({"answer": "fixed"})])
    assert groq_client(lambda r: next(replies)).extract("s", "u", Out).answer == "fixed"


def test_gemini_format_round_trips_through_openai_format():
    contents = [
        {"role": "user", "parts": [{"text": "python jobs"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "search_jobs", "args": {"query": "python"}}}]},
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "search_jobs", "response": {"result": {"count": 1}}}}],
        },
    ]
    msgs = to_openai_messages("sys", contents)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[3]["tool_call_id"] == msgs[2]["tool_calls"][0]["id"]
    tools = to_openai_tools(
        [{"name": "t", "parameters": {"type": "OBJECT", "properties": {"q": {"type": "STRING"}}}}]
    )
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {"q": {"type": "string"}}}
    back = from_openai_message(
        {"content": "Here", "tool_calls": [{"function": {"name": "t", "arguments": '{"q": 1}'}}]}
    )
    assert back["parts"] == [{"text": "Here"}, {"functionCall": {"name": "t", "args": {"q": 1}}}]


def test_router_falls_back_on_rate_limit_and_traces_both(records):
    groq = groq_client(lambda r: httpx.Response(429, json={"error": "rate limit reached"}))
    gem = gemini_client(lambda r: gemini_json_reply({"answer": "from gemini"}))
    router = RoutedLLM(
        [Route("groq", groq)],
        [Route("groq", groq), Route("gemini", gem)],
        [Route("groq", groq), Route("gemini", gem)],
    )
    assert router.extract("s", "u", Out).answer == "from gemini"
    assert [(r.provider, r.ok, r.fallback) for r in records] == [
        ("groq", False, False),
        ("gemini", True, True),
    ]
    assert records[1].task == "Out" and (records[1].tokens_in, records[1].tokens_out) == (20, 5)
    assert "429" in records[0].error


def test_router_does_not_hide_real_bugs(records):
    calls = []

    def bad_request(req):
        calls.append(1)
        return httpx.Response(400, json={"error": "invalid schema"})

    groq = groq_client(bad_request)
    gem = gemini_client(lambda r: gemini_json_reply({"answer": "x"}))
    router = RoutedLLM(
        [Route("groq", groq)],
        [Route("groq", groq), Route("gemini", gem)],
        [Route("groq", groq), Route("gemini", gem)],
    )
    with pytest.raises(LLMError, match="400"):
        router.extract("s", "u", Out)
    assert len(records) == 1  # Gemini was never tried


def test_router_sends_heavy_tasks_to_the_heavy_route(records):
    big = groq_client(lambda r: groq_tool_reply({"answer": "big"}))
    small = groq_client(lambda r: groq_tool_reply({"answer": "small"}))
    router = RoutedLLM([Route("groq", big)], [Route("groq", big)], [Route("groq", small)])
    assert router.extract("s", "u", ProjectIdeas).answer == "big"
    assert router.extract("s", "u", Out).answer == "small"


def test_retryable():
    assert retryable(LLMError("Groq 429: rate limit"))
    assert retryable(LLMError("Gemini timed out"))
    assert not retryable(LLMError("output failed validation twice"))
    assert not retryable(ValueError("429"))


def test_build_router_routes(monkeypatch):
    class S:
        groq_api_key, groq_model, groq_fast_model = "g", "big", "small"
        llm_api_key, llm_provider, llm_model = "m", "gemini", "flash"

    r = build_router(S())
    assert [x.provider for x in r.routes["chat"]] == ["groq", "gemini"]
    assert r.routes["light"][0].client.model_name == "small"
    byok = build_router(S(), user_gemini_key="mine")
    assert [x.provider for x in byok.routes["chat"]] == ["gemini-byok"]
    S.groq_api_key = S.llm_api_key = None
    assert build_router(S()) is None


def test_traces_are_stored_and_summarised(session):
    for provider, ok, ms in [
        ("groq", True, 100),
        ("groq", False, 300),
        ("gemini", True, 900),
        ("semantic-cache", True, 2),
    ]:
        session.execute(
            text(
                "INSERT INTO llm_calls (task, provider, model, ms, tokens_in, tokens_out, ok, fallback) VALUES ('chat', :p, 'm', :ms, 10, 5, :ok, :fb)"
            ),
            {"p": provider, "ms": ms, "ok": ok, "fb": provider == "gemini"},
        )
    s = trace.summary(session)
    assert s["llm_calls"] == 3 and s["errors"] == 1 and s["fallbacks"] == 1 and s["cache_hits"] == 1
    assert s["cache_hit_rate"] == 0.25 and s["tokens"] == 45


# --- semantic cache ------------------------------------------------------------------------------

from app.market.demand import canonical_role  # noqa: E402


@pytest.mark.parametrize(
    "a,b",
    [
        ("Front-End Developer", "frontend developer"),
        ("ML Engineer", "machine learning engineer"),
        ("SDE", "software development engineer"),
        ("Node.js Dev", "node developer"),
        ("SWE Internship", "software engineer intern"),
    ],
)
def test_canonical_role_matches_spellings(a, b):
    assert canonical_role(a) == canonical_role(b)


def test_canonical_role_keeps_different_roles_apart():
    assert canonical_role("java developer") != canonical_role("javascript developer")
    assert canonical_role("react developer") != canonical_role("react native developer")
