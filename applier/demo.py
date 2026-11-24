"""Auto-apply demo, on a mock job board only: shows how background auto-apply tools work.

    python -m applier.demo --profile applier/profiles/me.yaml --resume path/to/resume.pdf [--video out.mp4]

Pipeline (the same stages the commercial tools use):
  1. Discover: open the board and read every job.
  2. Filter: score fit from the job's skills vs the profile, skip senior roles and poor matches.
  3. Easy Apply driver: a state machine over the form's pages (contact, resume, questions, review).
  4. Question answering: answer bank first; open-ended questions from profile facts (or an LLM).
  5. Submit and track: submits on the MOCK board only, then shows the tracker.

This drives applier/demo_site, a local, fictional board. Real sites (LinkedIn included) forbid bots
in their terms; for them use `python -m applier apply`, which stops before Submit.
"""

import argparse
import functools
import http.server
import re
import shutil
import subprocess
import threading
from pathlib import Path

from applier.profile import Profile, load

SITE = Path(__file__).parent / "demo_site"
PREFERRED_CITIES = ("pune", "mumbai", "bengaluru", "bangalore", "remote")
SENIOR = re.compile(r"\b(senior|sr\.?|lead|architect|principal|manager)\b|\d+\+ years", re.I)


def serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(SITE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def fit(profile: Profile, job: dict) -> tuple[int, list[str], list[str]]:
    have = profile.all_skills() | {
        w.lower() for f in profile.summary_facts for w in re.findall(r"[A-Za-z+#.]+", f)
    }
    blob = " ".join(have) + " " + profile.source_text().lower()
    matched = [s for s in job["skills"] if s.lower() in blob]
    missing = [s for s in job["skills"] if s not in matched]
    return round(100 * len(matched) / max(1, len(job["skills"]))), matched, missing


def why_answer(profile: Profile, job: dict, matched: list[str]) -> str:
    """Open-ended question: built from TRUE profile facts that mention the job's skills."""
    facts = []
    for p in profile.projects:
        hits = [f for f in p.facts if any(s.lower() in f.lower() for s in matched)]
        if hits:
            facts.append((p.title.split(" -- ")[0], hits[0]))
    lines = [f"In {name}, I {fact[0].lower() + fact[1:]}." for name, fact in facts[:2]]
    return (
        " ".join(lines) or "My projects match what this role needs."
    ) + f" That is hands-on work with {', '.join(matched[:3])}, which this role uses daily."


def answer(profile: Profile, label: str, job: dict, matched: list[str]) -> str:
    low = label.lower()
    if "notice" in low:
        return "Immediately available"
    if "years of experience" in low:
        return "1" if any(s.lower() in low for s in matched) else "0"
    if "on-site" in low or "relocat" in low:
        return "Yes" if any(c in job["place"].lower() for c in PREFERRED_CITIES) else "No"
    if "stipend" in low or "salary" in low:
        return "As per company norms"
    if "why" in low or "fit" in low:
        return why_answer(profile, job, matched)
    return ""


def caption(page, text: str) -> None:
    page.evaluate(
        """t => { let d = document.getElementById('agent-cap'); if (!d) { d = document.createElement('div'); d.id = 'agent-cap';
          d.style.cssText = 'position:fixed;left:16px;bottom:16px;z-index:9999;max-width:620px;background:#14213D;color:#fff;font:600 15px system-ui;padding:10px 14px;border-radius:10px;box-shadow:0 6px 20px rgba(0,0,0,.3)';
          document.body.appendChild(d); } d.textContent = 'Agent: ' + t; }""",
        text,
    )


def run(
    profile: Profile, resume: Path, video: Path | None, headless: bool = True, min_fit: int = 50
) -> list[dict]:
    from playwright.sync_api import sync_playwright

    srv, base = serve()
    log = []
    tmp_video = Path("applier/out/demo_video")
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome", headless=headless, slow_mo=120)
        except Exception:  # noqa: BLE001
            browser = pw.chromium.launch(headless=headless, slow_mo=120)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 760},
            record_video_dir=str(tmp_video) if video else None,
            record_video_size={"width": 1280, "height": 760},
        )
        page = ctx.new_page()
        page.goto(f"{base}/index.html")
        page.evaluate("localStorage.clear()")
        page.reload()
        caption(page, "1. Discovering jobs on the board...")
        page.wait_for_timeout(1200)
        jobs = page.evaluate("window.JOBS")
        first, _, last = profile.contact.name.partition(" ")

        for job in jobs:
            page.click(f'[data-job="{job["id"]}"]')
            score, matched, missing = fit(profile, job)
            if SENIOR.search(job["title"] + " " + " ".join(job["skills"])):
                caption(page, f"2. {job['title']} at {job['company']}: senior role, skipping.")
                log.append(
                    {
                        "job": job["title"],
                        "company": job["company"],
                        "fit": score,
                        "status": "skipped (senior)",
                    }
                )
                page.wait_for_timeout(1400)
                continue
            if score < min_fit:
                caption(
                    page,
                    f"2. {job['title']} at {job['company']}: fit {score}% (missing {', '.join(missing)}), skipping.",
                )
                log.append(
                    {
                        "job": job["title"],
                        "company": job["company"],
                        "fit": score,
                        "status": "skipped (low fit)",
                    }
                )
                page.wait_for_timeout(1400)
                continue
            caption(
                page, f"2. {job['title']} at {job['company']}: fit {score}% ({', '.join(matched)}). Applying."
            )
            page.wait_for_timeout(1000)
            page.click("#easy-apply")
            # 3. Easy Apply state machine
            caption(page, "3. Page 1/4: contact details from the profile")
            page.fill("#fn", first)
            page.fill("#ln", last)
            page.fill("#em", profile.contact.email)
            page.fill("#ph", re.sub(r"^\+\d{1,3}\s*", "", profile.contact.phone))
            page.click("#next")
            caption(page, "3. Page 2/4: uploading the resume")
            page.set_input_files("#cv", str(resume))
            page.click("#next")
            caption(page, "4. Page 3/4: answering screening questions")
            for el in page.query_selector_all(".box input, .box select, .box textarea"):
                label = page.evaluate("e => e.labels[0] ? e.labels[0].innerText : ''", el)
                value = answer(profile, label, job, matched)
                if el.evaluate("e => e.tagName") == "SELECT":
                    el.select_option(label=value)
                else:
                    el.fill(value)
            page.wait_for_timeout(600)
            page.click("#next")
            caption(page, "5. Review page: everything filled. Submitting (mock board only).")
            page.wait_for_timeout(1500)
            page.click("#submit")
            page.wait_for_selector("#done")
            page.wait_for_timeout(900)
            page.click("#close")
            log.append({"job": job["title"], "company": job["company"], "fit": score, "status": "submitted"})

        page.goto(f"{base}/tracker.html")
        caption(
            page,
            f"Done: {sum(x['status'] == 'submitted' for x in log)} applied, {sum(x['status'] != 'submitted' for x in log)} skipped. All logged.",
        )
        page.wait_for_timeout(3000)
        vid = page.video
        ctx.close()
        browser.close()
        if video and vid:
            webm = Path(vid.path())
            video.parent.mkdir(parents=True, exist_ok=True)
            if shutil.which("ffmpeg"):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(webm),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(video),
                    ],
                    check=True,
                )
            else:
                shutil.copy(webm, video.with_suffix(".webm"))
    srv.shutdown()
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--video", help="Save a screen recording, e.g. applier/out/demo.mp4")
    ap.add_argument("--show", action="store_true", help="Show the browser window")
    a = ap.parse_args()
    for row in run(load(a.profile), Path(a.resume), Path(a.video) if a.video else None, headless=not a.show):
        print(f"{row['status']:<18} {row['fit']:>3}%  {row['job']} at {row['company']}")


if __name__ == "__main__":
    main()
