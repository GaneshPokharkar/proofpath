import shutil
from pathlib import Path

import pytest

from app.resume.tailor import Verdict, Verdicts
from applier.profile import Answers, Contact, Education, Experience, Profile, Project
from applier.tailor import (
    CoverLetter,
    JobInfo,
    Plan,
    Screen,
    build_packet,
    finish,
    keyword_coverage,
)

PROFILE = Profile(
    contact=Contact(
        name="Asha Kumar",
        email="asha@example.com",
        phone="+91 9000000000",
        city="Pune",
        postal_code="411001",
        github="https://github.com/asha",
    ),
    summary_facts=["B.Tech CSE student", "Builds and deploys web apps"],
    education=[Education(degree="B.Tech CSE", institution="VIT Pune", score="8.5", year="2028")],
    projects=[
        Project(
            id="shop",
            title="ShopKart -- E-commerce site",
            tech=["React", "Node.js", "MongoDB"],
            date="2025",
            facts=["Built a shopping site with React and Node.js", "Wrote unit tests with Jest"],
            bullets=["Built a shopping site with React and Node.js.", "Wrote unit tests with Jest."],
        ),
        Project(
            id="bot",
            title="StudyBot -- Telegram bot",
            tech=["Python"],
            date="2026",
            facts=["Python Telegram bot that sends study reminders"],
            bullets=["Python bot that sends study reminders."],
        ),
    ],
    experience=[
        Experience(
            id="intern",
            org="Acme",
            role="Web Intern",
            dates="2026",
            facts=["Fixed bugs in a Django site"],
            bullets=["Fixed bugs in a Django site."],
        )
    ],
    skills={"Languages": ["Python", "JavaScript"], "Tools": ["React", "Node.js", "Git"]},
    answers=Answers(how_heard="Job Board", notice_period="Immediate"),
)

JD = "Software Intern at Globex. Must know Java or JavaScript, React, testing. Nice: Kubernetes."


class FakeLLM:
    def __init__(self, plan: Plan):
        self.plan = plan

    def extract(self, system, user, schema):
        if schema is JobInfo:
            return JobInfo(
                title="Software Intern",
                company="Globex",
                must_have=["JavaScript", "React"],
                keywords=["Java/JavaScript", "React", "testing", "Kubernetes"],
            )
        if schema is Plan:
            return self.plan
        if schema is Verdicts:
            return Verdicts(verdicts=[Verdict(index=i, supported=True) for i in range(10)])
        if schema is Screen:
            return Screen(score=80, decision="shortlist", met=["React"], missing=["Kubernetes"], concerns=[])
        if schema is CoverLetter:
            return CoverLetter(text="I built ShopKart with React and Node.js. [AVAILABILITY]")
        raise AssertionError(schema)


def plan(**kw):
    base = dict(
        summary="B.Tech CSE student who builds and deploys web apps with React and Node.js.",
        projects=[
            {
                "id": "shop",
                "bullets": [
                    {
                        "text": "Built a React and Node.js shopping site with unit tests in Jest.",
                        "fact_numbers": [0, 1],
                    }
                ],
            }
        ],
        experience=[],
        skills_first=["React", "Kubernetes"],
    )
    base.update(kw)
    return Plan.model_validate(base)


def test_packet_uses_only_true_material():
    p = finish(FakeLLM(plan()), PROFILE, build_packet(FakeLLM(plan()), PROFILE, JD), JD)
    assert p.projects[0][1] == ["Built a React and Node.js shopping site with unit tests in Jest."]
    # Experience the plan skipped still appears, with its own bullets.
    assert p.experience[0][1] == ["Fixed bugs in a Django site."]
    # Kubernetes is asked for but not in the profile: never listed, reported as a gap.
    assert "kubernetes" not in str(p.skills).lower()
    assert list(p.skills)[0] == "Tools" and p.skills["Tools"][0] == "React"
    assert "Kubernetes" in p.gaps and "React" in p.covered and "Java/JavaScript" in p.covered
    assert p.screen.score == 80 and "[AVAILABILITY]" in p.cover_letter


def test_invented_claims_fall_back_to_the_profile_wording():
    bad = plan(
        projects=[
            {
                "id": "shop",
                "bullets": [
                    {"text": "Deployed ShopKart on Kubernetes for 10k users.", "fact_numbers": [0]},
                    {"text": "Wrote unit tests with Jest.", "fact_numbers": [1]},
                ],
            }
        ]
    )
    p = build_packet(FakeLLM(bad), PROFILE, JD)
    rejected = [ln for ln in p.lines if not ln.accepted]
    assert len(rejected) == 1 and "10k" in rejected[0].reason
    assert p.projects[0][1][0] == "Built a shopping site with React and Node.js."  # the profile's own bullet


def test_unknown_project_ids_are_ignored():
    p = build_packet(
        FakeLLM(plan(projects=[{"id": "nope", "bullets": [{"text": "x", "fact_numbers": []}]}])), PROFILE, JD
    )
    assert [proj.id for proj, _ in p.projects] == ["shop", "bot"]  # falls back to the profile's projects


@pytest.mark.parametrize(
    "keyword,covered",
    [
        ("C/C++", True),
        ("frontend", True),
        ("full-stack web development", True),
        ("cloud-native architectures", True),
        ("Java", False),
        ("distributed systems", False),
    ],
)
def test_keyword_coverage(keyword, covered):
    text = "Python, C++, C, JavaScript. JavaScript front end. full-stack web app. cloud-native app"
    got, missing = keyword_coverage(text, [keyword])
    assert (keyword in got) is covered


# --- the form filler, on a local page in a real headless browser ----------------------------------

FORM = """<!doctype html><html><body><form>
<label for=f>First Name *</label><input id=f required>
<label for=l>Last Name *</label><input id=l required>
<label for=e>Email</label><input id=e type=email>
<label for=p>Phone number</label><input id=p>
<label for=c>Country</label><select id=c><option>Please select</option><option>India</option><option>France</option></select>
<label for=h>How did you hear about us?</label><select id=h><option></option><option>Job Board</option><option>Referral</option></select>
<label for=n>Notice period</label><input id=n>
<label for=g>Gender</label><select id=g required><option></option><option>Female</option><option>Male</option></select>
<label><input type=checkbox id=a required> I agree to the privacy notice</label>
<label for=x>Years of Kubernetes experience *</label><input id=x required>
<label for=r>Resume/CV</label><input id=r type=file>
<label for=pre>City</label><input id=pre value="Mumbai">
<button type=submit id=submit>Submit</button></form>
<script>document.getElementById('submit').onclick = () => { document.title = 'SUBMITTED'; return false; }</script>
</body></html>"""


def _browser():
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    for kw in ({"channel": "chrome"}, {}, {"executable_path": shutil.which("google-chrome") or ""}):
        try:
            return pw, pw.chromium.launch(headless=True, **{k: v for k, v in kw.items() if v})
        except Exception:  # noqa: BLE001
            continue
    pw.stop()
    pytest.skip("no browser available")


def test_fill_form_fills_known_fields_and_never_consents_or_submits(tmp_path):
    from applier.browser import fill_form

    pdf = tmp_path / "Asha_Kumar_Resume.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    page_file = tmp_path / "form.html"
    page_file.write_text(FORM)
    pw, browser = _browser()
    try:
        page = browser.new_page()
        page.goto(page_file.as_uri())
        rep = fill_form(page, PROFILE, Path(pdf), cover_letter="")
        vals = dict(rep.filled)
        assert vals["First Name *"] == "Asha" and vals["Last Name *"] == "Kumar"
        assert vals["Email"] == "asha@example.com" and vals["Phone number"] == "9000000000"
        assert vals["Country"] == "India" and vals["How did you hear about us?"] == "Job Board"
        assert vals["Notice period"] == "Immediate"
        assert page.input_value("#pre") == "Mumbai"  # never overwrites a filled field
        assert rep.uploaded and page.eval_on_selector("#r", "e => e.files[0].name") == "Asha_Kumar_Resume.pdf"
        assert not page.is_checked("#a")  # never consents
        assert page.eval_on_selector("#g", "e => e.selectedIndex") == 0  # never answers gender
        assert any("Gender" in s for s in rep.skipped_sensitive) and any(
            "privacy" in s for s in rep.skipped_sensitive
        )
        assert "Years of Kubernetes experience *" in rep.still_required
        assert page.title() != "SUBMITTED"  # never submits
    finally:
        browser.close()
        pw.stop()
