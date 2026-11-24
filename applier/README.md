# Applier: tailor a resume for a job and fill its application

```bash
pip install -r requirements.txt playwright      # uses your installed Chrome; else: playwright install chromium
cp applier/profiles/example.yaml applier/profiles/me.yaml   # fill with your TRUE facts (not committed)
export GROQ_API_KEY=...            # and/or JOBPILOT_LLM_API_KEY (Gemini) as the fallback

python -m applier run --profile applier/profiles/me.yaml --job https://company.com/jobs/123
```

**What happens**

1. **Read the job** in a real browser (career sites render with JavaScript), or from `--jd-file`.
2. **Tailor** from your master profile: pick the most relevant projects, rewrite your own bullets in
   the job's words, order skills by what the job asks for.
3. **Fact-check** every new line (rule checks, then an LLM judge against your profile). Anything
   your profile doesn't back is rejected and your own wording is used instead.
4. **One-page PDF** in your layout (single column, clean text for ATS parsers), trimmed to fit.
5. **Report** (`applier/out/<company>-<role>/report.md`): an AI recruiter-style screen score,
   job keywords covered, and honest gaps to learn or build rather than claim. Plus a cover letter.
6. **Fill the form** in a visible browser: press Apply, you sign in if asked, it uploads the PDF and
   fills empty fields it recognises (name, email, phone, city, links, saved answers).
7. **You submit.** It never ticks consent or declaration boxes, never answers gender, disability or
   salary questions, never solves CAPTCHAs and never clicks Submit. It lists what it filled and what
   is still required, waits for you, and logs the application in `applier/out/applications.csv`.

Every model call is traced to `applier/out/llm_calls.jsonl` (task, provider, latency, tokens).

## Demo: how background auto-apply tools work (mock board only)

```bash
python -m applier.demo --profile applier/profiles/me.yaml --resume me.pdf --video applier/out/demo.mp4 [--show]
```

Runs a fully autonomous agent against `applier/demo_site`, a local job board with fictional jobs:
discover jobs → score fit from skills (skip poor fits and senior roles) → drive the multi-page
Easy Apply form → answer screening questions from the answer bank and true profile facts → submit →
log to the tracker. It records the run as a video. It never targets real sites; on real boards the
platform terms forbid bots, so `python -m applier apply` stops before Submit instead.
