"""Auto job applier: tailor a fact-checked resume for a job, then fill its application form.

    python -m applier prepare --profile applier/profiles/me.yaml --job https://.../job/123
    python -m applier apply   --profile applier/profiles/me.yaml --job https://.../job/123 [--packet DIR]
    python -m applier run     --profile applier/profiles/me.yaml --job https://.../job/123

prepare  reads the job, writes applier/out/<company>-<role>/: resume.pdf, resume.tex,
         cover_letter.txt and report.md (screen score, keywords covered and honest gaps,
         every rewrite the fact-checker rejected).
apply    opens the job in a visible browser, presses Apply, uploads resume.pdf, fills what it can
         and stops. You review and press Submit yourself.
run      both, pausing between them so you can read the report.

--jd-file PATH uses a saved job description instead of reading the job page.
Needs GROQ_API_KEY and/or JOBPILOT_LLM_API_KEY (Gemini) in the environment or .env.
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).parent / "out"


def _slug(*parts: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", " ".join(parts).lower()).strip("-")[:60] or "job"


def prepare(args) -> Path:
    from app.config import get_settings
    from app.llmops.router import build_router
    from applier.profile import load
    from applier.render import build_pdf
    from applier.tailor import build_packet, finish

    profile = load(args.profile)
    _trace_to_folder()
    if args.jd_file:
        jd = Path(args.jd_file).read_text(encoding="utf-8")
    else:
        from playwright.sync_api import sync_playwright

        from applier.browser import launch, read_job_page

        with sync_playwright() as pw:
            ctx = launch(pw, headless=True)
            jd = read_job_page(ctx.new_page(), args.job)
            ctx.close()
    llm = build_router(get_settings())
    if llm is None:
        sys.exit("Set GROQ_API_KEY or JOBPILOT_LLM_API_KEY first.")
    print("Reading the job and tailoring the resume...")
    packet = build_packet(llm, profile, jd)
    packet = finish(llm, profile, packet, jd)
    source, pdf, dropped = build_pdf(profile, packet, get_settings().latex_compile_url)

    folder = OUT / _slug(packet.job.company, packet.job.title)
    folder.mkdir(parents=True, exist_ok=True)
    first = profile.contact.name.split()[0]
    (folder / f"{profile.contact.name.replace(' ', '_')}_Resume.pdf").write_bytes(pdf)
    (folder / "resume.tex").write_text(source, encoding="utf-8")
    (folder / "job_description.txt").write_text(jd, encoding="utf-8")
    if packet.cover_letter:
        (folder / "cover_letter.txt").write_text(packet.cover_letter, encoding="utf-8")
    s = packet.screen
    rejected = [ln for ln in packet.lines if not ln.accepted]
    report = [
        f"# {packet.job.title} at {packet.job.company}",
        f"Prepared {datetime.now():%Y-%m-%d %H:%M} for {first}. Job: {args.job or args.jd_file}",
        "",
        f"## Screen: {s.score}/100, {s.decision}" if s else "## Screen: not run",
        *(
            [
                "",
                "Met: " + "; ".join(s.met),
                "Missing: " + ("; ".join(s.missing) or "none"),
                "Concerns: " + ("; ".join(s.concerns) or "none"),
            ]
            if s
            else []
        ),
        "",
        "## Job keywords",
        "Covered: " + ", ".join(packet.covered),
        "Not on the resume (the profile doesn't back them; learn or build them, don't claim them): "
        + (", ".join(packet.gaps) or "none"),
        "",
        f"## Fact-checker: {len(packet.lines) - len(rejected)} lines kept, {len(rejected)} rejected",
        *[f"- REJECTED ({ln.where}): {ln.text}\n  reason: {ln.reason}" for ln in rejected],
        *(["", "## Cut to fit one page", *[f"- {d}" for d in dropped]] if dropped else []),
        "",
        "Before applying: fill [AVAILABILITY] in cover_letter.txt and read the resume once.",
    ]
    (folder / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:12]))
    print(f"\nPacket: {folder}")
    return folder


def _trace_to_folder() -> None:
    """On a laptop there is no ProofPath database: keep the LLM traces in applier/out/llm_calls.jsonl."""
    import json
    from dataclasses import asdict

    from app.llmops import trace

    OUT.mkdir(exist_ok=True)

    def sink(rec):
        with (OUT / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec) | {"at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}) + "\n")

    trace.sink = sink


def apply(args, folder: Path | None = None) -> None:
    from playwright.sync_api import sync_playwright

    from applier.browser import fill_form, launch, open_application
    from applier.profile import load

    profile = load(args.profile)
    folder = folder or Path(args.packet)
    pdf = next(folder.glob("*_Resume.pdf"))
    letter_path = folder / "cover_letter.txt"
    letter = letter_path.read_text(encoding="utf-8") if letter_path.exists() else ""
    if "[AVAILABILITY]" in letter:
        print("Note: cover_letter.txt still has [AVAILABILITY]; it won't be pasted into forms.")
        letter = ""
    with sync_playwright() as pw:
        ctx = launch(pw)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        open_application(page, args.job)
        input(
            "\nIf the site asks you to sign in or create an account, do it in the browser now.\n"
            "When the application form (with the resume upload) is on screen, press Enter here... "
        )
        rep = fill_form(page, profile, pdf, letter)
        print(f"\nResume uploaded: {'yes' if rep.uploaded else 'no (upload it yourself)'}")
        print("Filled:")
        for label, value in rep.filled:
            print(f"  - {label[:50]}: {value}")
        if rep.skipped_sensitive:
            print("Left for you (consents, declarations, personal questions):")
            for label in rep.skipped_sensitive:
                print(f"  - {label[:80]}")
        if rep.still_required:
            print("Required and still empty:")
            for label in rep.still_required:
                print(f"  - {label[:80]}")
        print("\nNothing has been submitted. Check every page, then press Submit in the browser yourself.")
        done = (
            input("Type 'submitted' once you have submitted (anything else to skip logging): ")
            .strip()
            .lower()
        )
        _log(folder, args.job, done == "submitted")
        ctx.close()


def _log(folder: Path, job: str, submitted: bool) -> None:
    path = OUT / "applications.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "job", "packet", "status"])
        w.writerow(
            [f"{datetime.now():%Y-%m-%d %H:%M}", job, folder.name, "submitted" if submitted else "prepared"]
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m applier", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["prepare", "apply", "run"])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--job", help="Job page URL")
    ap.add_argument("--jd-file", help="Job description saved as text, instead of reading the page")
    ap.add_argument("--packet", help="Folder made by prepare (for apply)")
    args = ap.parse_args()
    if args.command in ("prepare", "run") and not (args.job or args.jd_file):
        ap.error("--job or --jd-file is required")
    if args.command == "apply" and not (args.packet and args.job):
        ap.error("apply needs --packet and --job")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "apply":
        apply(args)
    else:
        folder = prepare(args)
        input(f"\nRead {folder / 'report.md'} and the PDF. Press Enter to open the application... ")
        apply(args, folder)


if __name__ == "__main__":
    main()
