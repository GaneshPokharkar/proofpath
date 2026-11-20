"""Post-deploy smoke test: hit the live endpoints a student uses and fail loudly if any breaks.

    python scripts/smoke.py https://proofpath-....vercel.app [--commit SHA] [--wait 600]

With --commit it first waits until /version reports that commit (Vercel builds take a minute or
two), so the checks run against the new deployment, not the old one. Uses no AI and no secrets.
"""

import argparse
import sys
import time
import uuid

import httpx

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def health(c):
    assert c.get("/health").json() == {"status": "ok"}


@check
def home_page(c):
    r = c.get("/")
    assert r.status_code == 200 and "ProofPath" in r.text and 'id="chat-fab"' in r.text


@check
def search_india(c):
    d = c.get("/search", params={"q": "software engineer", "india": "true", "limit": 5}).json()
    assert d["count"] > 0, "no India jobs returned"


@check
def search_filters(c):
    d = c.get("/search", params={"q": "intern", "level": "intern", "field": "tech", "limit": 5}).json()
    assert all(r["level"] == "intern" for r in d["results"])


@check
def fields(c):
    assert len(c.get("/fields").json()["fields"]) >= 5


@check
def job_detail(c):
    jid = c.get("/search", params={"q": "python", "limit": 1}).json()["results"][0]["id"]
    assert c.get(f"/jobs/{jid}").json()["id"] == jid


@check
def stats(c):
    d = c.get("/stats").json()
    assert d["active_jobs"] > 100, f"only {d['active_jobs']} active jobs"


@check
def chat_memory_roundtrip(c):
    sid = "smoke" + uuid.uuid4().hex[:20]
    assert c.get(f"/chat/{sid}/memory").json()["turns"] == 0
    assert c.delete(f"/chat/{sid}").json() == {"forgotten": True}


@check
def admin_is_locked(c):
    assert c.post("/admin/refresh").status_code in (401, 503)
    assert c.get("/cron/daily").status_code == 401


def wait_for_commit(c, sha: str, seconds: int) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            live = c.get("/version").json().get("commit", "")
            if live and (live.startswith(sha) or sha.startswith(live)):
                print(f"live commit {live[:7]}")
                return
            print(f"waiting: live {live[:7] or '?'}, want {sha[:7]}")
        except (httpx.HTTPError, ValueError) as exc:
            print(f"waiting: {exc}")
        time.sleep(15)
    raise SystemExit(f"commit {sha[:7]} did not go live within {seconds}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--commit")
    ap.add_argument("--wait", type=int, default=600)
    a = ap.parse_args()
    with httpx.Client(base_url=a.url.rstrip("/"), timeout=60, follow_redirects=True) as c:
        if a.commit:
            wait_for_commit(c, a.commit, a.wait)
        failed = 0
        for fn in CHECKS:
            try:
                fn(c)
                print(f"ok    {fn.__name__}")
            except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
                failed += 1
                print(f"FAIL  {fn.__name__}: {exc}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
