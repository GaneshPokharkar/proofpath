# 1. LLM observability: you can't improve what you can't see

## The problem
An LLM call is a slow, paid, unreliable network call that returns text you can't unit-test
exactly. In production you need to answer: which feature makes the most calls? Which model is
slow? How often does a provider fail? How many tokens (money) does a resume check cost? Before
this, ProofPath couldn't answer any of these, and we only found out the free Gemini quota had run
out when a feature stopped working.

## The design
- **One record per call** (`app/llmops/trace.py`, `CallRecord`): task, provider, model, latency
  in ms, input and output tokens, success, error, whether it was a fallback.
- **Captured in one place.** Every call goes through the router (`app/llmops/router.py`), which
  wraps each attempt in `timed(...)`. Features never think about tracing.
- **Task = schema name.** `extract(..., schema=ResumeProfile)` is traced as task `ResumeProfile`,
  so the dashboard groups calls by feature with no extra code.
- **Cache hits are traced too** (provider `cache` / `semantic-cache`), so the hit rate is real.
- **Privacy:** no prompt or answer text is stored. A resume never lands in the traces table.
- **Never breaks a request:** the sink swallows its own errors. Tests swap the sink for a list.
- **Retention:** traces are deleted after 30 days.

## Read the code
1. `app/llmops/trace.py`: `CallRecord`, `timed()`, `summary()` (p50/p95 with `percentile_disc`).
2. `app/llmops/router.py`, `RoutedLLM._run`: where every attempt is timed and recorded.
3. `app/llm.py`, `GeminiLLM._post`: where token usage is read from the response.
4. `GET /stats/llm` and the "AI calls" table on `/admin`.

## Check it
- `pytest tests/test_llmops.py -k trace`
- Open `/admin` after using the site: calls, errors, fallbacks, p50/p95, tokens per task.

## Interview questions
**Why p95 and not the average latency?** Averages hide the slow tail that users feel. If 1 in 20
chat replies takes 12 s, the average can still look fine; p95 shows it.

**What would you trace in a RAG or agent system?** Per step: retrieval latency and number of
documents, each tool call and its error, model and tokens, total turn time, and a trace id tying
the steps of one request together.

**Why not log prompts?** They hold personal data (resumes), are large, and logging them creates
a privacy liability. Log shape and numbers; sample full prompts only with consent or redaction.

**How would you add cost?** Multiply tokens by each model's price per million tokens, kept in a
config table because prices change. Alert when daily cost or tokens per request jumps.

**Langfuse / LangSmith vs your own table?** Hosted tools add prompt playgrounds, trace trees and
evals. A table is enough to start and shows you understand what they record; exporting to
Langfuse later is a sink swap.
