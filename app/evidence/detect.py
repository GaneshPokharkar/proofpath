"""Deterministic skill evidence from a repo snapshot: no LLM, every claim points at a file.

Rules of thumb that keep this honest:
- a language counts only if it is a real share of the code (not three lines of a script);
- a dependency counts because someone declared it in a manifest, which is weaker than proof
  of use, so the detail string always names the manifest line it came from;
- forks are never evidence: the code is someone else's.
"""

import json
import re
import tomllib
from dataclasses import dataclass

from app.evidence.github import RepoSnapshot

MIN_LANGUAGE_SHARE = 0.05
MIN_LANGUAGE_BYTES = 2_000

LANGUAGE_SKILLS: dict[str, tuple[str, str]] = {
    "Python": ("Python", "language"),
    "JavaScript": ("JavaScript", "language"),
    "TypeScript": ("TypeScript", "language"),
    "Java": ("Java", "language"),
    "Go": ("Go", "language"),
    "C++": ("C++", "language"),
    "C": ("C", "language"),
    "C#": ("C#", "language"),
    "Rust": ("Rust", "language"),
    "Kotlin": ("Kotlin", "language"),
    "Swift": ("Swift", "language"),
    "Dart": ("Dart", "language"),
    "Ruby": ("Ruby", "language"),
    "PHP": ("PHP", "language"),
    "Scala": ("Scala", "language"),
    "R": ("R", "language"),
    "PLpgSQL": ("SQL", "language"),
    "Verilog": ("Verilog", "language"),
    "SystemVerilog": ("SystemVerilog", "language"),
    "VHDL": ("VHDL", "language"),
    "MATLAB": ("MATLAB", "language"),
    "HCL": ("Terraform", "devops"),
}

DEPENDENCY_SKILLS: dict[str, tuple[str, str]] = {
    # Python
    "fastapi": ("FastAPI", "framework"),
    "flask": ("Flask", "framework"),
    "django": ("Django", "framework"),
    "sqlalchemy": ("SQLAlchemy", "framework"),
    "alembic": ("SQLAlchemy", "framework"),
    "psycopg": ("PostgreSQL", "database"),
    "psycopg2": ("PostgreSQL", "database"),
    "psycopg2-binary": ("PostgreSQL", "database"),
    "asyncpg": ("PostgreSQL", "database"),
    "pgvector": ("pgvector", "database"),
    "pymongo": ("MongoDB", "database"),
    "redis": ("Redis", "database"),
    "torch": ("PyTorch", "ml"),
    "tensorflow": ("TensorFlow", "ml"),
    "keras": ("Keras", "ml"),
    "scikit-learn": ("scikit-learn", "ml"),
    "sklearn": ("scikit-learn", "ml"),
    "pandas": ("pandas", "tool"),
    "numpy": ("NumPy", "tool"),
    "transformers": ("Hugging Face", "ml"),
    "sentence-transformers": ("Embeddings", "ml"),
    "fastembed": ("Embeddings", "ml"),
    "faiss-cpu": ("FAISS", "ml"),
    "chromadb": ("ChromaDB", "database"),
    "langchain": ("LangChain", "framework"),
    "llama-index": ("LlamaIndex", "framework"),
    "openai": ("LLMs", "ml"),
    "anthropic": ("LLMs", "ml"),
    "google-genai": ("LLMs", "ml"),
    "google-generativeai": ("LLMs", "ml"),
    "opencv-python": ("OpenCV", "ml"),
    "opencv-python-headless": ("OpenCV", "ml"),
    "ultralytics": ("YOLO", "ml"),
    "streamlit": ("Streamlit", "framework"),
    "pytest": ("pytest", "tool"),
    "celery": ("Celery", "tool"),
    "boto3": ("AWS", "cloud"),
    "pydantic": ("Pydantic", "framework"),
    # JavaScript / TypeScript
    "react": ("React", "framework"),
    "next": ("Next.js", "framework"),
    "vue": ("Vue.js", "framework"),
    "express": ("Express", "framework"),
    "@nestjs/core": ("NestJS", "framework"),
    "mongoose": ("MongoDB", "database"),
    "mongodb": ("MongoDB", "database"),
    "pg": ("PostgreSQL", "database"),
    "prisma": ("Prisma", "tool"),
    "@prisma/client": ("Prisma", "tool"),
    "tailwindcss": ("Tailwind CSS", "framework"),
    "typescript": ("TypeScript", "language"),
    "react-native": ("React Native", "framework"),
    "socket.io": ("WebSockets", "concept"),
    "jest": ("Jest", "tool"),
    "@tensorflow/tfjs": ("TensorFlow", "ml"),
    "openai-node": ("LLMs", "ml"),
    "@anthropic-ai/sdk": ("LLMs", "ml"),
    "@google/genai": ("LLMs", "ml"),
}


@dataclass(frozen=True)
class Evidence:
    skill: str
    category: str
    source: str  # language | dependency | file
    detail: str


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalise_pkg(name: str) -> str:
    return re.sub(r"[_.]+", "-", name.strip().lower())


def python_dependencies(files: dict[str, str]) -> list[tuple[str, str]]:
    """(package, where) pairs from requirements.txt and pyproject.toml."""
    found = []
    for line in files.get("requirements.txt", "").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-") and (m := _REQ_NAME.match(line)):
            found.append((_normalise_pkg(m.group(1)), f"requirements.txt: {line[:60]}"))
    if "pyproject.toml" in files:
        try:
            data = tomllib.loads(files["pyproject.toml"])
        except tomllib.TOMLDecodeError:
            data = {}
        for spec in (data.get("project") or {}).get("dependencies") or []:
            if m := _REQ_NAME.match(spec):
                found.append((_normalise_pkg(m.group(1)), f"pyproject.toml: {spec[:60]}"))
        poetry = ((data.get("tool") or {}).get("poetry") or {}).get("dependencies") or {}
        for name in poetry:
            if name.lower() != "python":
                found.append((_normalise_pkg(name), f"pyproject.toml: {name}"))
    return found


def node_dependencies(files: dict[str, str]) -> list[tuple[str, str]]:
    if "package.json" not in files:
        return []
    try:
        data = json.loads(files["package.json"])
    except json.JSONDecodeError:
        return []
    found = []
    for section in ("dependencies", "devDependencies"):
        for name, version in (data.get(section) or {}).items():
            found.append((name.lower(), f"package.json {section}: {name}@{version}"))
    return found


def detect(snapshot: RepoSnapshot) -> list[Evidence]:
    if snapshot.is_fork:
        return []
    evidence: dict[tuple[str, str], Evidence] = {}

    def add(skill: str, category: str, source: str, detail: str):
        evidence.setdefault((skill, source), Evidence(skill, category, source, detail))

    total = sum(snapshot.languages.values()) or 1
    for lang, size in snapshot.languages.items():
        if lang in LANGUAGE_SKILLS and (size / total >= MIN_LANGUAGE_SHARE and size >= MIN_LANGUAGE_BYTES):
            skill, category = LANGUAGE_SKILLS[lang]
            add(skill, category, "language", f"{lang}: {size / total:.0%} of code")

    for pkg, where in python_dependencies(snapshot.files) + node_dependencies(snapshot.files):
        if pkg in DEPENDENCY_SKILLS:
            skill, category = DEPENDENCY_SKILLS[pkg]
            add(skill, category, "dependency", where)

    if "Dockerfile" in snapshot.files:
        add("Docker", "devops", "file", "Dockerfile")
    for compose in ("docker-compose.yml", "docker-compose.yaml", "compose.yaml"):
        if compose in snapshot.files:
            add("Docker Compose", "devops", "file", compose)
    if snapshot.workflows:
        names = ", ".join(snapshot.workflows[:3])
        add("GitHub Actions", "devops", "file", f".github/workflows: {names}")
        add("CI/CD", "devops", "file", f".github/workflows: {names}")
    return list(evidence.values())
