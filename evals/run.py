"""Run the 100-case eval set against the real chat agent and fail if quality regresses.

Metrics
- tool_choice_accuracy: the agent calls the expected tool in that turn ("none" = calls no tool).
- filter_extraction_accuracy: on search cases, every expected filter (field, level, remote,
  location) is present in the search_jobs call with the right value.
- hallucinated_job_rate: share of job ids cited in replies that no tool returned in that turn
  (the agent strips them before the student sees them; this measures how often it tries).

Usage: python -m evals.run [--limit N] [--only TAG] [--out evals/report.json]
Needs JOBPILOT_DATABASE_URL (seeded from evals/jobs.json automatically) and JOBPILOT_LLM_API_KEY.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from sqlalchemy import text

from app.agent import memory
from app.agent.agent import run_turn
from app.agent.tools import ToolContext
from app.config import get_settings
from app.db import session_scope
from app.embeddings import FastEmbedEmbedder
from app.llm import GeminiChat
from evals.seed import seed

HERE = Path(__file__).parent
THRESHOLDS = json.loads((HERE / "thresholds.json").read_text())
REF = re.compile(r"\{job:(\d+)\}")


def job_ids(session) -> list[int]:
    return [r[0] for r in session.execute(text("SELECT id FROM jobs WHERE is_active ORDER BY id")).all()]


def check_args(expect: dict, args: dict) -> bool:
    for k, v in expect.items():
        got = args.get(k)
        if k == "location":
            if not got or v.lower() not in str(got).lower():
                return False
        elif got != v:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only")
    ap.add_argument("--out", default=str(HERE / "report.json"))
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between cases (free-tier rate limits)")
    a = ap.parse_args()

    s = get_settings()
    if not s.llm_api_key:
        print("JOBPILOT_LLM_API_KEY is not set", file=sys.stderr)
        return 2
    model = GeminiChat(s.llm_api_key, s.llm_model)
    embedder = FastEmbedEmbedder(s.embedding_model)
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines() if line.strip()]
    if a.only:
        cases = [c for c in cases if c["tag"] == a.only]
    cases = cases[: a.limit] if a.limit else cases

    with session_scope() as session:
        if session.execute(text("SELECT count(*) FROM jobs")).scalar() < 50:
            print("seeding", seed(session, embedder), "fixture jobs")
        ids = job_ids(session)

    results, cited_total, hallucinated_total = [], 0, 0
    for i, case in enumerate(cases, 1):
        fill = lambda m: str(ids[int(m.group(1)) % len(ids)])  # noqa: E731
        message = REF.sub(fill, case["message"])
        with session_scope() as session:
            mem = memory.Memory(f"eval_{case['id']}_{int(time.time())}")
            mem.profile = dict(case["setup"].get("profile", {}))
            if "shortlist" in case["setup"]:
                mem.profile["shortlist"] = [ids[n % len(ids)] for n in case["setup"]["shortlist"]]
            ctx = ToolContext(session=session, embedder=embedder, llm=model, mem=mem)
            started = time.perf_counter()
            try:
                out = run_turn(model, ctx, message)
                calls, reply, removed = out.tool_calls, out.reply, out.removed_job_ids
                cited = len(re.findall(r"\[job:\d+\]", reply)) + len(removed)
                error = None
            except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a crashed run
                calls, reply, removed, cited, error = [], "", [], 0, f"{type(exc).__name__}: {exc}"[:300]
            session.rollback()  # evals never change the database
        first = calls[0]["name"] if calls else "none"
        names = [c["name"] for c in calls]
        # Right if the expected tool is called in the turn (saving a preference first is fine);
        # "none" means no tool may be called.
        tool_ok = not names if case["expect_tool"] == "none" else case["expect_tool"] in names
        search_call = next((c for c in calls if c["name"] == "search_jobs"), None)
        args_ok = None
        if case["tag"] == "search":
            args_ok = bool(search_call) and check_args(case["expect_args"], search_call["args"])
        elif case["expect_args"]:
            target = next((c for c in calls if c["name"] == case["expect_tool"]), None)
            args_ok = bool(target) and check_args(case["expect_args"], target["args"])
        cited_total += cited
        hallucinated_total += len(removed)
        results.append(
            {
                "id": case["id"],
                "tag": case["tag"],
                "message": message,
                "expected": case["expect_tool"],
                "tools": [{"name": c["name"], "args": c["args"]} for c in calls],
                "tool_ok": tool_ok,
                "args_ok": args_ok,
                "hallucinated": removed,
                "reply": reply[:400],
                "error": error,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        mark = "ok " if tool_ok and args_ok is not False else "BAD"
        print(
            f"[{i:3}/{len(cases)}] {mark} {case['id']} {case['tag']:<10} {message[:60]!r} -> {first}",
            flush=True,
        )
        time.sleep(a.pause)

    n = len(results)
    search = [r for r in results if r["tag"] == "search"]
    with_args = [r for r in results if r["args_ok"] is not None]
    metrics = {
        "cases": n,
        "tool_choice_accuracy": round(sum(r["tool_ok"] for r in results) / n, 3),
        "filter_extraction_accuracy": round(sum(r["args_ok"] for r in search) / len(search), 3)
        if search
        else None,
        "argument_accuracy_all": round(sum(r["args_ok"] for r in with_args) / len(with_args), 3)
        if with_args
        else None,
        "hallucinated_job_rate": round(hallucinated_total / cited_total, 3) if cited_total else 0.0,
        "errors": sum(1 for r in results if r["error"]),
        "by_tag": {
            t: round(
                sum(r["tool_ok"] for r in results if r["tag"] == t)
                / max(1, sum(r["tag"] == t for r in results)),
                3,
            )
            for t in sorted({r["tag"] for r in results})
        },
        "model": s.llm_model,
    }
    Path(a.out).write_text(json.dumps({"metrics": metrics, "results": results}, indent=1))
    print(json.dumps(metrics, indent=1))

    failed = []
    if metrics["tool_choice_accuracy"] < THRESHOLDS["tool_choice_accuracy_min"]:
        failed.append("tool choice")
    if (
        metrics["filter_extraction_accuracy"] is not None
        and metrics["filter_extraction_accuracy"] < THRESHOLDS["filter_extraction_accuracy_min"]
    ):
        failed.append("filter extraction")
    if metrics["hallucinated_job_rate"] > THRESHOLDS["hallucinated_job_rate_max"]:
        failed.append("hallucinated jobs")
    if metrics["errors"] > THRESHOLDS["errors_max"]:
        failed.append("errors")
    if failed:
        print(
            "REGRESSION:", ", ".join(failed), "below the thresholds in evals/thresholds.json", file=sys.stderr
        )
        return 1
    print("All eval thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
