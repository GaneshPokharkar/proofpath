"""One-off: snapshot ~120 real postings from the live ProofPath API into evals/jobs.json, so the
eval suite runs against a fixed, realistic job set (same jobs every run, no live scraping)."""

import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://proofpath-ganeshpokharkars-projects.vercel.app"
QUERIES = [
    ("python developer", "software", "intern"),
    ("data analyst", "data_ai", None),
    ("machine learning", "data_ai", "intern"),
    ("react frontend", "frontend", None),
    ("devops kubernetes", "devops_cloud", None),
    ("security engineer", "security", None),
    ("android ios", "mobile", None),
    ("qa automation test", "qa", None),
    ("product manager", "product", None),
    ("designer ux", "design", None),
    ("backend java", "software", None),
    ("data engineer spark", "data_ai", None),
    ("software engineer", "software", "entry"),
    ("full stack", "software", None),
    ("sales marketing", "other", None),
    ("ai research", "data_ai", None),
    ("cloud aws", "devops_cloud", None),
    ("intern", "tech", "intern"),
]
c = httpx.Client(timeout=60)
seen, out = set(), []
for q, field, level in QUERIES:
    params = {"q": q, "limit": 10, "field": field}
    if level:
        params["level"] = level
    for hit in c.get(f"{BASE}/search", params=params).json()["results"]:
        if hit["id"] in seen:
            continue
        seen.add(hit["id"])
        j = c.get(f"{BASE}/jobs/{hit['id']}").json()
        out.append(
            {
                k: j.get(k)
                for k in ("title", "company", "location", "remote", "url", "description", "posted_at")
            }
            | {"external_id": str(hit["id"]), "source": hit["source"], "apply_by": hit.get("apply_by")}
        )
with open("evals/jobs.json", "w") as f:
    json.dump(out, f, indent=1)
print(len(out), "jobs")
