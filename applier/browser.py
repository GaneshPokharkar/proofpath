"""Browser side: read a job page, and fill an application form up to (never including) Submit.

Runs a real, visible browser with its own saved profile (applier/.browser), so the candidate signs
in to job sites once and stays signed in. The filler:
- uploads the tailored PDF to the resume field,
- waits for the site's own resume parsing to fill what it can,
- fills only EMPTY fields whose labels it recognises, from the profile and its saved answers,
- never ticks consent or declaration boxes, never solves CAPTCHAs, never clicks Submit,
- lists every field it filled and every required field still empty, then waits for the candidate
  to review and submit in the browser themselves.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from applier.profile import Profile

BROWSER_DIR = Path(__file__).parent / ".browser"

# Collect every visible form control with the best label we can find, and tag it for later.
_SCAN_JS = r"""
() => {
  const vis = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return (r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none') || e.type === 'file'; };
  const labelOf = e => {
    let t = e.getAttribute('aria-label') || '';
    if (!t && e.labels && e.labels.length) t = [...e.labels].map(l => l.innerText).join(' ');
    if (!t && e.getAttribute('aria-labelledby')) t = e.getAttribute('aria-labelledby').split(' ').map(i => document.getElementById(i)?.innerText || '').join(' ');
    if (!t) { const wrap = e.closest('label, [class*=field], [class*=Field], [class*=question], div'); if (wrap) t = (wrap.querySelector('label, legend, [class*=label], [class*=Label]')?.innerText || ''); }
    return (t || e.placeholder || e.name || e.id || '').replace(/\s+/g, ' ').trim();
  };
  const out = [];
  document.querySelectorAll('input, select, textarea').forEach((e, i) => {
    if (['hidden', 'submit', 'button', 'image', 'reset'].includes(e.type) || !vis(e) || e.disabled) return;
    e.setAttribute('data-applier', String(i));
    out.push({ key: String(i), tag: e.tagName.toLowerCase(), type: e.type || '', label: labelOf(e),
      required: e.required || e.getAttribute('aria-required') === 'true' || /\*\s*$/.test(labelOf(e)),
      value: e.type === 'checkbox' || e.type === 'radio' ? (e.checked ? 'on' : '') : (e.tagName === 'SELECT' ? (e.selectedIndex > 0 ? e.options[e.selectedIndex].text : '') : e.value),
      options: e.tagName === 'SELECT' ? [...e.options].map(o => o.text) : [] });
  });
  return out;
}
"""

# Fields the filler must never touch: consents, declarations, and sensitive self-identification.
NEVER = re.compile(
    r"consent|agree|privacy|terms|declar|certify|acknowledg|gender|sex\b|race|ethnic|disab|veteran|religio|caste|"
    r"marital|date of birth|\bdob\b|nationality|salary|ctc|stipend|password|captcha|aadhaar|pan\b|passport",
    re.I,
)


def answers_for(profile: Profile, cover_letter: str) -> list[tuple[re.Pattern, str]]:
    c, a = profile.contact, profile.answers
    first, _, last = c.name.partition(" ")
    rules = [
        (r"^(legal |given )?first name|given name|forename", first),
        (r"last name|family name|surname", last),
        (r"^(full |your )?name\b|candidate name", c.name),
        (r"e-?mail", c.email),
        (r"phone|mobile|contact number", re.sub(r"^\+\d{1,3}\s*", "", c.phone)),
        (r"\bcity\b|town", c.city),
        (r"postal|zip|pin ?code", c.postal_code),
        (r"\bstate\b|province|region", c.state),
        (r"\bcountry\b", c.country),
        (r"linkedin", c.linkedin),
        (r"github", c.github),
        (r"website|portfolio|personal (site|url)", c.website or c.github),
        (r"how did you (hear|find)|source", a.how_heard),
        (r"notice period", a.notice_period),
        (r"available|start date|joining", a.available_from),
        (r"duration", a.internship_duration),
        (r"relocat", a.willing_to_relocate),
        (r"authori[sz]", a.work_authorization),
        (r"cover letter|why (do you want|are you interested)|motivation", cover_letter),
    ]
    rules += [(re.escape(k), v) for k, v in a.extra.items()]
    return [(re.compile(p, re.I), v) for p, v in rules if v]


@dataclass
class FillReport:
    filled: list[tuple[str, str]]
    skipped_sensitive: list[str]
    still_required: list[str]
    uploaded: bool


def _pick_option(options: list[str], want: str) -> str | None:
    w = want.lower()
    for o in options:
        if o.lower() == w:
            return o
    for o in options:
        if w in o.lower() or (o.lower() and o.lower() in w):
            return o
    return None


def fill_form(page, profile: Profile, resume_pdf: Path, cover_letter: str = "") -> FillReport:
    fields = page.evaluate(_SCAN_JS)
    uploaded = False
    files = [f for f in fields if f["type"] == "file"]
    if files:
        target = next((f for f in files if re.search(r"resume|cv", f["label"], re.I)), files[0])
        page.set_input_files(f'[data-applier="{target["key"]}"]', str(resume_pdf))
        uploaded = True
        page.wait_for_timeout(6000)  # let the site parse the resume and autofill
        fields = page.evaluate(_SCAN_JS)

    rules = answers_for(profile, cover_letter)
    filled, sensitive = [], []
    for f in fields:
        if f["type"] in ("file", "checkbox", "radio"):
            if NEVER.search(f["label"]):
                sensitive.append(f["label"])
            continue
        if NEVER.search(f["label"]):
            sensitive.append(f["label"])
            continue
        if f["value"]:
            continue  # never overwrite what the site or the candidate already put there
        for pattern, value in rules:
            if not pattern.search(f["label"]):
                continue
            sel = f'[data-applier="{f["key"]}"]'
            if f["tag"] == "select":
                opt = _pick_option(f["options"], value)
                if opt is None:
                    break
                page.select_option(sel, label=opt)
                filled.append((f["label"], opt))
            else:
                page.fill(sel, value)
                filled.append((f["label"], value if len(value) < 60 else value[:57] + "..."))
            break

    after = page.evaluate(_SCAN_JS)
    still = [f["label"] for f in after if f["required"] and not f["value"] and f["type"] != "file"]
    return FillReport(filled, sorted(set(sensitive)), still, uploaded)


def launch(playwright, headless: bool = False):
    """A persistent, visible browser so sign-ins survive between runs. Uses installed Chrome when
    available (no extra download), else Playwright's Chromium."""
    BROWSER_DIR.mkdir(exist_ok=True)
    kw = dict(user_data_dir=str(BROWSER_DIR), headless=headless, viewport={"width": 1280, "height": 900})
    try:
        return playwright.chromium.launch_persistent_context(channel="chrome", **kw)
    except Exception:  # noqa: BLE001 - Chrome not installed
        return playwright.chromium.launch_persistent_context(**kw)


def read_job_page(page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(4000)  # many career sites render the description with JavaScript
    return page.evaluate("() => document.body.innerText")[:20000]


def open_application(page, job_url: str) -> None:
    """Go to the job and press its Apply button if there is one (the candidate signs in if asked)."""
    page.goto(job_url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(3000)
    btn = page.get_by_role("link", name=re.compile(r"^\s*apply", re.I)).or_(
        page.get_by_role("button", name=re.compile(r"^\s*apply", re.I))
    )
    if btn.count():
        btn.first.click()
        page.wait_for_timeout(5000)
