"""Model router: pick a model per task, fall back to the next provider when one fails, and trace
every attempt.

Routes (first that answers wins):
  chat            -> Groq large model  -> Gemini
  heavy extract   -> Groq large model  -> Gemini   (resume parsing, rewrites, fact-checks)
  light extract   -> Groq small model  -> Gemini   (job skills, summaries, short profiles)

Only errors that another provider can fix cause a fallback: rate limits and quota (429),
overload and server errors (5xx), timeouts. A schema-validation failure or a bad request is a
bug in our prompt or schema, so it is raised, not hidden by trying elsewhere.
"""

import re
from dataclasses import dataclass

from app.llm import GeminiChat, LLMError, T
from app.llmops.trace import CallRecord, timed

# Extraction schemas that need the larger model: long inputs and careful judgement.
HEAVY = {"ResumeDoc", "Rewrites", "Verdicts", "ResumeEdits", "ProjectIdeas", "ResumeProfile"}
_RETRYABLE = re.compile(r"\b(429|500|502|503|504)\b|timed out|quota|rate", re.I)


def retryable(exc: Exception) -> bool:
    return isinstance(exc, LLMError) and bool(_RETRYABLE.search(str(exc)))


@dataclass
class Route:
    provider: str  # label for traces: "groq", "gemini", "gemini-byok"
    client: object  # has extract(); chat() too for chat routes


class RoutedLLM:
    """Looks like one model to the app (extract + chat), routes inside."""

    def __init__(self, chat: list[Route], heavy: list[Route], light: list[Route]):
        if not (chat and heavy and light):
            raise LLMError("every route needs at least one provider")
        self.routes = {"chat": chat, "heavy": heavy, "light": light}
        self.model_name = chat[0].client.model_name

    def _run(self, tier: str, task: str, call):
        errors = []
        for i, route in enumerate(self.routes[tier]):
            rec = CallRecord(
                task=task, provider=route.provider, model=route.client.model_name, fallback=i > 0
            )
            try:
                with timed(rec):
                    out = call(route.client)
                    rec.tokens_in, rec.tokens_out = getattr(route.client, "last_usage", (0, 0))
                return out
            except LLMError as exc:
                errors.append(f"{route.provider}: {exc}")
                if not retryable(exc) or i == len(self.routes[tier]) - 1:
                    raise LLMError(" | ".join(errors)[:900]) from exc

    def extract(self, system: str, user: str, schema: type[T]) -> T:
        tier = "heavy" if schema.__name__ in HEAVY else "light"
        return self._run(tier, schema.__name__, lambda c: c.extract(system, user, schema))

    def chat(self, system: str, contents: list[dict], tools: list[dict], final: bool = False) -> dict:
        return self._run("chat", "chat", lambda c: c.chat(system, contents, tools, final=final))


def build_router(settings, user_gemini_key: str | None = None) -> RoutedLLM | None:
    """Server routes from the configured keys. A visitor's own Gemini key gets a Gemini-only route
    (their key, their quota). None when no provider is configured."""
    if user_gemini_key:
        g = Route("gemini-byok", GeminiChat(user_gemini_key, settings.llm_model, max_retries=1))
        return RoutedLLM([g], [g], [g])
    groq_big = groq_small = None
    if settings.groq_api_key:
        from app.llmops.groq import GroqLLM

        groq_big = Route("groq", GroqLLM(settings.groq_api_key, settings.groq_model))
        groq_small = Route("groq", GroqLLM(settings.groq_api_key, settings.groq_fast_model))
    gemini = None
    if settings.llm_api_key and settings.llm_provider == "gemini":
        # With a Groq route in front, fail over fast instead of waiting out Gemini's backoff.
        gemini = Route(
            "gemini", GeminiChat(settings.llm_api_key, settings.llm_model, max_retries=1 if groq_big else 4)
        )
    big = [r for r in (groq_big, gemini) if r]
    small = [r for r in (groq_small, gemini) if r]
    if not big:
        return None
    return RoutedLLM(big, big, small)
