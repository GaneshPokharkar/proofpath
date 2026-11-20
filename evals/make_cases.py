"""Writes evals/cases.jsonl: 100 fixed cases for the chat agent. Run once; the file is committed.

Each case: message, optional setup (profile / shortlist), the tool the agent should call first
("none" = answer without tools), and for searches the filters it must extract.
Job references like {job:N} are filled with the Nth job of the fixture at run time."""

import json
import random
from pathlib import Path

random.seed(7)
cases = []


def add(msg, tool, args=None, setup=None, tag=""):
    cases.append(
        {
            "id": f"c{len(cases) + 1:03d}",
            "tag": tag,
            "message": msg,
            "expect_tool": tool,
            "expect_args": args or {},
            "setup": setup or {},
        }
    )


# A. Search with filters (50) -------------------------------------------------------------------
roles = [  # (words in the message, field the agent must pick or None if several are reasonable)
    ("data science", "data_ai"),
    ("machine learning", "data_ai"),
    ("data analyst", "data_ai"),
    ("AI", "data_ai"),
    ("devops", "devops_cloud"),
    ("cloud engineering", "devops_cloud"),
    ("cyber security", "security"),
    ("android developer", "mobile"),
    ("iOS", "mobile"),
    ("QA testing", "qa"),
    ("UI/UX design", "design"),
    ("product manager", "product"),
    ("python developer", None),
    ("backend", None),
    ("react", None),
    ("full stack", None),
    ("java developer", None),
]
levels = [
    ("internships", "intern"),
    ("intern roles", "intern"),
    ("fresher jobs", "entry"),
    ("entry level jobs", "entry"),
    ("senior roles", "senior"),
    ("jobs", None),
]
places = [
    ("in Bangalore", "bangalore"),
    ("in Pune", "pune"),
    ("in London", "london"),
    ("in Berlin", "berlin"),
    ("remote", "REMOTE"),
    ("", None),
    ("", None),
    ("in New York", "new york"),
]
templates = [
    "{role} {level} {place}",
    "show me {role} {level} {place}",
    "any {role} {level} {place}?",
    "I'm looking for {role} {level} {place}",
    "find {role} {level} {place} please",
]
for _ in range(50):
    (rw, field), (lw, level), (pw, place) = random.choice(roles), random.choice(levels), random.choice(places)
    msg = " ".join(random.choice(templates).format(role=rw, level=lw, place=pw).split())
    args = {}
    if field:
        args["field"] = field
    if level:
        args["level"] = level
    if place == "REMOTE":
        args["remote"] = True
    elif place:
        args["location"] = place
    add(msg, "search_jobs", args, tag="search")

# B. Remembering preferences (15) ---------------------------------------------------------------
prefs = [
    ("I know Python, SQL and Tableau", "skills"),
    ("My skills are React and Node.js", "skills"),
    ("I have worked with Docker and Kubernetes", "skills"),
    ("I only want to work in Pune", "locations"),
    ("I prefer jobs in Hyderabad or Chennai", "locations"),
    ("I want remote work only", "remote"),
    ("Please remember I'm looking for internships", "levels"),
    ("I am not interested in sales roles", "rejected_roles"),
    ("Never show me QA testing jobs", "rejected_roles"),
    ("I need at least 6 LPA", "salary_floor"),
    ("My expected salary floor is 50k per month", "salary_floor"),
    ("I'm interested in data and AI roles", "fields"),
    ("Remember that I know Java and Spring Boot", "skills"),
    ("I'd like to stay in Mumbai", "locations"),
    ("I don't want any senior positions", "rejected_roles"),
]
for msg, key in prefs:
    add(msg, "save_preference", {"key": key}, tag="preference")

# C. Shortlist (10) -----------------------------------------------------------------------------
for i, tmpl in enumerate(
    [
        "save job {job:%d} to my shortlist",
        "shortlist job {job:%d}",
        "add {job:%d} to my saved jobs",
        "please bookmark job {job:%d}",
        "keep job {job:%d} for later",
    ]
):
    add(tmpl % (i * 7), "shortlist", {"action": "add"}, tag="shortlist")
add(
    "remove job {job:3} from my shortlist",
    "shortlist",
    {"action": "remove"},
    setup={"shortlist": [3]},
    tag="shortlist",
)
add(
    "drop {job:10} from my saved list",
    "shortlist",
    {"action": "remove"},
    setup={"shortlist": [10]},
    tag="shortlist",
)
for msg in ["show my shortlist", "what jobs have I saved?", "list my saved jobs"]:
    add(msg, "shortlist", {"action": "list"}, setup={"shortlist": [1, 2]}, tag="shortlist")

# D. Compare (8) --------------------------------------------------------------------------------
for a, b in [(1, 2), (4, 9), (12, 20), (5, 30), (40, 41), (8, 15), (22, 60), (70, 90)]:
    msg = random.choice(
        [
            "compare job {job:%d} and job {job:%d}",
            "which is better, {job:%d} or {job:%d}?",
            "how do jobs {job:%d} and {job:%d} differ?",
        ]
    ) % (a, b)
    add(msg, "compare_jobs", tag="compare")

# E. Explain a match (8) ------------------------------------------------------------------------
profile = {"skills": ["Python", "SQL", "Pandas"], "locations": ["Bangalore"]}
for n in [1, 6, 11, 16, 25, 33, 47, 55]:
    msg = (
        random.choice(
            [
                "is job {job:%d} a good fit for me?",
                "why would job {job:%d} suit me?",
                "how well do I match job {job:%d}?",
            ]
        )
        % n
    )
    add(msg, "explain_match", setup={"profile": profile}, tag="explain")

# F. Job details (5) ----------------------------------------------------------------------------
for n in [2, 13, 21, 34, 50]:
    add(
        random.choice(
            [
                "tell me more about job {job:%d}",
                "what does job {job:%d} involve?",
                "details of job {job:%d} please",
            ]
        )
        % n,
        "get_job",
        tag="details",
    )

# G. No tool needed (4) -------------------------------------------------------------------------
for msg in ["hi", "what can you do?", "thanks, that's all", "how do I use this?"]:
    add(msg, "none", tag="chitchat")

assert len(cases) == 100, len(cases)
Path(__file__).with_name("cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
print(len(cases), "cases")
