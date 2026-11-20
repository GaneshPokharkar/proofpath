from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.agent import memory
from app.agent.agent import run_turn
from app.agent.tools import ToolContext
from app.api.main import app, get_chat_model
from app.db import get_session
from app.ingest.pipeline import sync_postings
from tests.test_pipeline_and_search import FakeSource, posting

SID = "testsession_abcdefgh"


class ScriptedChat:
    """Plays back model turns. Each step is a list of parts; a str becomes a text part, a tuple
    (name, args) a function call. Callables get the contents so far and return parts."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.seen_systems = []
        self.contents = []

    def chat(self, system, contents, tools, final=False):
        self.seen_systems.append(system)
        self.contents = contents
        step = self.steps.pop(0)
        if callable(step):
            step = step(contents)
        parts = [
            {"text": p} if isinstance(p, str) else {"functionCall": {"name": p[0], "args": p[1]}}
            for p in step
        ]
        return {"role": "model", "parts": parts}

    def extract(self, system, user, schema):  # rolling summary
        return schema(summary="Student wants python internships in Pune.")


@pytest.fixture
def jobs(session, embedder):
    src = FakeSource()
    postings = [
        posting("p1", "Python Developer Intern", "Python, Django and SQL.", location="Pune"),
        posting("p2", "Data Analyst Intern", "SQL, Excel and Tableau.", location="Bangalore"),
        posting("p3", "Senior Java Engineer", "Java and Spring.", location="Pune"),
    ]
    sync_postings(session, src, postings, embedder, datetime.now(UTC), timedelta(days=14))
    ids = dict(session.execute(text("SELECT external_id, id FROM jobs")).all())
    return ids


def ctx(session, embedder, mem=None):
    return ToolContext(session=session, embedder=embedder, llm=None, mem=mem or memory.Memory(SID))


def first_job_id(contents):
    resp = contents[-1]["parts"][0]["functionResponse"]["response"]["result"]
    return resp["jobs"][0]["id"]


def test_search_then_cite_only_real_jobs(session, embedder, jobs):
    model = ScriptedChat(
        [("search_jobs", {"query": "python intern", "level": "intern", "location": "Pune"})],
        lambda c: [f"Here is one: [job:{first_job_id(c)}] and [job:999999]"],
    )
    c = ctx(session, embedder)
    out = run_turn(model, c, "python internships in pune")
    assert [t["name"] for t in out.tool_calls] == ["search_jobs"]
    assert out.removed_job_ids == [999999]  # hallucinated id stripped
    assert "999999" not in out.reply
    assert out.jobs[0]["title"] == "Python Developer Intern"


def test_bad_tool_args_go_back_to_the_model(session, embedder, jobs):
    model = ScriptedChat(
        [("get_job", {"job_id": 424242})],
        lambda c: [
            "That job doesn't exist."
            if "error" in c[-1]["parts"][0]["functionResponse"]["response"]["result"]
            else "?"
        ],
    )
    out = run_turn(model, ctx(session, embedder), "show job 424242")
    assert out.tool_calls[0]["error"].startswith("There is no job")
    assert out.reply == "That job doesn't exist."


def test_preferences_shortlist_compare_and_explain(session, embedder, jobs):
    c = ctx(session, embedder)
    from app.agent import tools

    tools.save_preference(c, "skills", "Python, SQL")
    tools.save_preference(c, "locations", "Pune")
    tools.save_preference(c, "rejected_roles", "senior")
    assert c.mem.profile["skills"] == ["Python", "SQL"]
    found = tools.search_jobs(c, "engineer developer")
    assert all("Senior" not in j["title"] for j in found["jobs"])  # rejected roles are filtered
    tools.shortlist(c, "add", jobs["p1"])
    tools.shortlist(c, "add", jobs["p1"])
    assert c.mem.shortlist == [jobs["p1"]]
    assert tools.shortlist(c, "list")["shortlist"][0]["id"] == jobs["p1"]
    from app.market.skills import SkillResolver

    resolver = SkillResolver(session, embedder)
    resolver.resolve("Python", "language")
    resolver.resolve("Django", "framework")
    session.execute(
        text(
            "INSERT INTO job_skills (job_id, skill_id, importance) SELECT :j, id, CASE name WHEN 'Python' THEN 'required' ELSE 'preferred' END FROM skills WHERE name IN ('Python','Django')"
        ),
        {"j": jobs["p1"]},
    )
    fit = tools.explain_match(c, jobs["p1"])
    assert fit["matched_skills"] == ["Python"] and fit["missing_preferred"] == ["Django"]
    assert fit["skill_match"] == round(1 / 1.5, 2)
    cmp = tools.compare_jobs(c, [jobs["p1"], jobs["p2"]])
    assert cmp["jobs"][0]["student_has_required"] == ["Python"]
    with pytest.raises(tools.ToolError):
        tools.compare_jobs(c, [jobs["p1"]])


def test_two_layer_memory_persists_and_folds(session, embedder, jobs):
    model = ScriptedChat(
        *([[("save_preference", {"key": "skills", "value": "Python"})], ["Saved."]] + [["ok"]] * 7)
    )
    from app.agent.agent import chat

    chat(model, model, session, embedder, SID, "I know Python")
    mem = memory.load(session, SID)
    assert mem.profile["skills"] == ["Python"] and mem.turns == 1
    for i in range(7):
        chat(model, model, session, embedder, SID, f"message {i}")
    mem = memory.load(session, SID)
    assert len(mem.recent) == memory.KEEP_TURNS and mem.summary.startswith("Student wants")
    assert '"skills": ["Python"]' in model.seen_systems[-1]  # profile injected every turn
    assert "Summary of the earlier conversation" in model.seen_systems[-1]
    memory.forget(session, SID)
    assert memory.load(session, SID).turns == 0


def test_chat_endpoint(session, embedder, jobs):
    model = ScriptedChat(
        [("search_jobs", {"query": "data analyst"})], lambda c: [f"Try [job:{first_job_id(c)}]."]
    )
    app.state.embedder = embedder
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_chat_model] = lambda: model
    try:
        with TestClient(app) as client:
            r = client.post("/chat", json={"session_id": SID, "message": "data analyst jobs"})
            assert r.status_code == 200, r.text
            assert r.json()["tools"] == ["search_jobs"] and r.json()["jobs"]
            assert client.get(f"/chat/{SID}/memory").json()["turns"] == 1
            assert client.post("/chat", json={"session_id": "bad", "message": "x"}).status_code == 422
            assert client.delete(f"/chat/{SID}").json() == {"forgotten": True}
    finally:
        app.dependency_overrides.clear()
        app.state.embedder = None
