"""Field and level of a job, from its title. Rule-based on purpose: titles are short and
formulaic, the rules are instant and free, and every decision can be read and tested.

Field order matters: the first matching rule wins, so specific fields come before the general
"software" catch-all, and clearly non-tech titles are caught before anything else.
"""

import re

# (field, label shown to students, title patterns)
FIELDS: list[tuple[str, str, str]] = [
    (
        "other",
        "Non-tech",
        r"\b(sales|account (executive|manager)|marketing|recruit|talent|hr\b|human resources|"
        r"finance|financial|accountant|accounting|workplace|people (partner|operations)|legal|counsel|paralegal|customer success|customer support|support specialist|"
        r"office|executive assistant|business development|bdr|sdr|partnerships?|communications|content writer|"
        r"copywriter|procurement|payroll|tax\b|audit|compliance officer|operations manager|facilities)\b",
    ),
    (
        "data_ai",
        "Data & AI",
        r"\b(data|machine learning|ml|ai|artificial intelligence|deep learning|nlp|llm|"
        r"computer vision|analytics|(business|bi|product|reporting|insights) analyst|scientist|research engineer|"
        r"mlops)\b",
    ),
    (
        "security",
        "Security",
        r"\b(security|appsec|infosec|penetration|pentest|soc analyst|threat|cryptograph|"
        r"vulnerability|detection engineer)",
    ),
    (
        "devops_cloud",
        "Cloud & DevOps",
        r"\b(devops|sre|site reliability|platform engineer|cloud (engineer|architect|"
        r"infrastructure|operations|ops|platform)|infrastructure|"
        r"kubernetes|systems engineer|network engineer|release engineer|build engineer)\b",
    ),
    ("mobile", "Mobile", r"\b(android|ios|mobile|flutter|react native|kotlin|swift)\b"),
    (
        "frontend",
        "Frontend & Web",
        r"\b(front[- ]?end|frontend|ui engineer|web developer|web engineer|javascript|"
        r"react developer|angular|vue)\b",
    ),
    ("qa", "QA & Testing", r"\b(qa|quality assurance|test|testing|sdet|automation engineer)\b"),
    ("design", "Design", r"\b(designer|ux|ui/ux|product design|user research)\b"),
    (
        "product",
        "Product",
        r"\b(product manager|product owner|program manager|technical program|tpm|scrum)\b",
    ),
    (
        "software",
        "Software Engineering",
        r"\b(engineer|engineering|developer|programmer|sde|swe|backend|back[- ]end|"
        r"full[- ]?stack|software|architect|technical lead|tech lead|firmware|embedded)\b",
    ),
]

# Senior is checked before entry so "Senior Associate" or "Associate Director" are not entry level.
LEVELS: list[tuple[str, str, str]] = [
    (
        "intern",
        "Internship",
        r"\b(intern|internship|trainee|apprentice|working student|werkstudent|praktikum|"
        r"co-?op|summer)\b",
    ),
    (
        "senior",
        "Senior",
        r"\b(senior|sr\.?|staff|principal|lead|manager|director|head|vp|vice president|"
        r"architect|distinguished|fellow|chief|ii|iii|iv|sde[- ]?(2|3|ii|iii)|l[4-7])\b",
    ),
    (
        "entry",
        "Entry level / fresher",
        r"\b(junior|jr\.?|entry|graduate|new grad|fresher|early career|"
        r"associate (software|engineer|developer|data|analyst|consultant)|engineer i|developer i|sde[- ]?(i|1)|"
        r"swe[- ]?(i|1)|analyst i|level 1|l1)\b",
    ),
]

_FIELD_RE = [(key, re.compile(p, re.IGNORECASE)) for key, _, p in FIELDS]
_LEVEL_RE = [(key, re.compile(p, re.IGNORECASE)) for key, _, p in LEVELS]

FIELD_LABELS = {key: label for key, label, _ in FIELDS}
LEVEL_LABELS = {key: label for key, label, _ in LEVELS} | {"mid": "Mid level"}
TECH_FIELDS = tuple(k for k, _, _ in FIELDS if k != "other")


_TECH_ROLE = re.compile(
    r"\b(engineer|developer|programmer|scientist|sde|swe|devops|designer)\b", re.IGNORECASE
)


def classify_field(title: str) -> str:
    for key, pattern in _FIELD_RE:
        if key == "other" and _TECH_ROLE.search(title):
            continue  # "Front End Engineer, Marketing" is an engineering job
        if pattern.search(title):
            return key
    return "other"


def classify_level(title: str) -> str:
    for key, pattern in _LEVEL_RE:
        if pattern.search(title):
            return key
    return "mid"
