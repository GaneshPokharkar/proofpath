"""The chat agent: an LLM that answers job-search questions by calling the six tools.

One turn: the model sees the system prompt (with the student's long-term profile and the rolling
summary), the last few turns, and the new message. It may call tools up to MAX_STEPS times; each
result goes back to it. The final text may cite jobs only as [job:ID] for ids a tool returned in
this turn or the shortlist; any other id is removed before the reply is shown, and counted.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.agent import memory as memory_mod
from app.agent.tools import DECLARATIONS, TOOLS, ToolContext, ToolError

MAX_STEPS = 6
JOB_REF = re.compile(r"\[job:(\d+)\]")

SYSTEM = """You are ProofPath, a job-search assistant for students and freshers, mostly in India.
You answer with the tools; never invent a job, company, link or deadline.
- To find jobs, call search_jobs. Turn what the student says into filters: field (data_ai, software,
  devops_cloud, security, mobile, frontend, qa, design, product, other, or tech), level (intern,
  entry, mid, senior), remote, location. Put only the role and skills in query.
- Use the saved profile below to fill filters the student did not repeat (e.g. saved locations).
- When the student tells you their skills, places, level, remote wish, salary floor or roles they
  don't want, call save_preference so you remember it.
- "Save / shortlist / remember this job" -> shortlist add; "my shortlist" -> shortlist list.
- "Compare these" -> compare_jobs with the ids. "Is this a fit / why this job" -> explain_match.
- Refer to a job only as [job:ID] with an id a tool gave you. The app shows the job card there.
- Be brief and friendly: a sentence or two, then the jobs. No markdown tables.
- If a search finds nothing, retry once with fewer filters (drop location, then level); then say
  what you tried and suggest a change. Ask a question only when you must."""


class ChatModel(Protocol):
    def chat(self, system: str, contents: list[dict], tools: list[dict], final: bool = False) -> dict: ...


@dataclass
class TurnResult:
    reply: str
    jobs: list[dict]  # cards for the jobs cited in the reply, in order
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "args", "error"?}]
    removed_job_ids: list[int] = field(default_factory=list)  # cited but never returned: hallucinated


def _history(mem: memory_mod.Memory) -> list[dict]:
    out = []
    for t in mem.recent:
        out.append({"role": "user", "parts": [{"text": t["user"]}]})
        out.append({"role": "model", "parts": [{"text": t["agent"]}]})
    return out


def run_turn(model: ChatModel, ctx: ToolContext, message: str) -> TurnResult:
    system = f"{SYSTEM}\n\n{ctx.mem.for_prompt()}"
    contents = _history(ctx.mem) + [{"role": "user", "parts": [{"text": message}]}]
    calls: list[dict] = []
    reply = ""
    for step in range(MAX_STEPS):
        content = model.chat(system, contents, DECLARATIONS, final=step == MAX_STEPS - 1)
        fcalls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
        if not fcalls:
            reply = "".join(p.get("text", "") for p in content["parts"]).strip()
            break
        contents.append(content)
        responses = []
        for fc in fcalls:
            name, args = fc.get("name"), dict(fc.get("args") or {})
            record = {"name": name, "args": args}
            try:
                if name not in TOOLS:
                    raise ToolError(f"Unknown tool {name}")
                result = TOOLS[name](ctx, **args)
            except (ToolError, TypeError) as exc:
                record["error"] = str(exc)
                result = {"error": str(exc)}
            calls.append(record)
            responses.append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": {"result": json.loads(json.dumps(result, default=str))},
                    }
                }
            )
        contents.append({"role": "user", "parts": responses})
    else:
        reply = "I couldn't finish that. Try asking in a simpler way."

    allowed = set(ctx.seen)
    removed = [int(i) for i in JOB_REF.findall(reply) if int(i) not in allowed]
    if removed:
        reply = JOB_REF.sub(lambda m: m.group(0) if int(m.group(1)) in allowed else "", reply)
    cited = list(dict.fromkeys(int(i) for i in JOB_REF.findall(reply)))
    return TurnResult(reply or "Done.", [ctx.seen[i] for i in cited], calls, removed)


def chat(model, structured_llm, session, embedder, session_id: str, message: str) -> TurnResult:
    """Load memory, run one turn, store both memory layers."""
    mem = memory_mod.load(session, session_id)
    ctx = ToolContext(session=session, embedder=embedder, llm=structured_llm, mem=mem)
    result = run_turn(model, ctx, message)
    memory_mod.add_turn(structured_llm, mem, message, result.reply)
    memory_mod.save(session, mem)
    return result
