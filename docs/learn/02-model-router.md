# 2. Model routing and fallback: no single provider is reliable or cheap enough

## The problem
On 20 November the free Gemini key hit its daily limit (500 requests) and every AI feature
stopped. One provider is a single point of failure, and using the biggest model for every small
task wastes quota and time.

## The design (`app/llmops/router.py`)
| Task | First choice | Fallback |
|---|---|---|
| Chat agent (tool calling) | Groq, Llama 3.3 70B | Gemini |
| Heavy extraction: resume parsing, rewrites, fact-checks, project ideas | Groq, Llama 3.3 70B | Gemini |
| Light extraction: job skills, conversation summaries | Groq, Llama 3.1 8B (fast) | Gemini |

- **Same interface everywhere.** `RoutedLLM` has `extract()` and `chat()`, like a single model.
  Features don't know which provider answered.
- **Fall back only when another provider can help:** 429 (rate limit or quota), 5xx, timeouts.
  A validation failure or 400 is our bug and is raised, not hidden (`retryable()`).
- **Fail over fast.** With Groq in front, Gemini's own backoff is shortened, so a student waits
  seconds, not a minute.
- **Provider adapter** (`app/llmops/groq.py`): the agent speaks Gemini's message format; the Groq
  client translates messages, tool declarations and tool calls to the OpenAI format and back.
- **Bring your own key:** a visitor's own Gemini key gets a Gemini-only route on their quota.

## Read the code
1. `RoutedLLM._run`: the loop over providers, tracing each attempt.
2. `to_openai_messages` / `from_openai_message`: how function calls are paired with results.
3. `build_router`: how routes are assembled from the keys that exist.

## Check it
- `pytest tests/test_llmops.py`: fallback on 429, no fallback on 400, heavy vs light routing,
  message translation round trip.
- `/admin` shows the `fallback` count and which provider served each task.

## Interview questions
**Why route by task instead of always using the best model?** Cost and latency scale with model
size. Simple extraction is as accurate on a small model; reserve the large one for reasoning and
long context. Measure with evals before moving a task down.

**What can go wrong with fallback?** Different models behave differently, so quality can silently
drop on the fallback. Track fallbacks, and run evals against every model in a route. Also: if
both providers share a cause (bad input), retrying wastes time, which is why only some errors fall
back.

**Why not retry the same provider?** Per-minute limits recover in seconds (a short retry is in
the client), but a daily quota won't recover for hours; switching providers is the fix.

**How would you add a circuit breaker?** After N failures in a window, skip that provider for a
cooldown instead of paying a timeout on every request; probe it occasionally to close the circuit.
