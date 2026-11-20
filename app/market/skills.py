"""Extract skills from job postings and normalise their names.

Normalisation, in order:
1. exact alias match ("postgres" -> PostgreSQL), from SEED_ALIASES plus every spelling seen before;
2. embedding match to an existing skill of the same category above MERGE_THRESHOLD;
3. otherwise a new canonical skill.
Each new spelling is stored as an alias, so step 1 handles it for free next time.
"""

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.embeddings import Embedder
from app.llm import StructuredLLM
from app.models import Job
from app.models_skills import JobSkill, JobSkillExtraction, Skill, SkillAlias

Category = Literal["language", "framework", "database", "cloud", "ml", "devops", "tool", "concept"]

# Measured with bge-small on skill names: wrong pairs can score higher than true variants
# (Java~JavaScript 0.83, Docker~Docker Compose 0.86, but scikit-learn~sklearn 0.83,
# Amazon S3~AWS S3 0.87), so no threshold separates them. 0.93 only merges near-identical
# spellings (Next.js~NextJS 0.93); known variants go in SEED_ALIASES and versions are stripped.
MERGE_THRESHOLD = 0.93
_VERSION_SUFFIX = re.compile(r"\s+v?\d+(\.\d+)*(\.x)?\+?$", re.IGNORECASE)
MAX_DESCRIPTION_CHARS = 6000

SEED_ALIASES: dict[str, str] = {
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "psql": "PostgreSQL",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "google cloud": "Google Cloud",
    "aws": "AWS",
    "amazon web services": "AWS",
    "golang": "Go",
    "go": "Go",
    "react.js": "React",
    "reactjs": "React",
    "react": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "llm": "LLMs",
    "llms": "LLMs",
    "large language models": "LLMs",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "sklearn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "airflow": "Apache Airflow",
    "apache airflow": "Apache Airflow",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "pyspark": "Apache Spark",
    "kafka": "Apache Kafka",
    "apache kafka": "Apache Kafka",
    "s3": "Amazon S3",
    "aws s3": "Amazon S3",
    "amazon s3": "Amazon S3",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "next": "Next.js",
    "torch": "PyTorch",
    "pytorch": "PyTorch",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    "hf transformers": "Hugging Face",
    "rest": "REST APIs",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "restful apis": "REST APIs",
}


class ExtractedSkill(BaseModel):
    name: str = Field(description="Common canonical name, e.g. 'PostgreSQL', 'React', 'Kubernetes'")
    category: Category
    importance: Literal["required", "preferred"]


class ExtractedSkills(BaseModel):
    skills: list[ExtractedSkill] = Field(max_length=30)


SYSTEM_PROMPT = """You extract the technical skills a job posting asks for.
Rules:
- Only concrete, learnable technical skills: languages, frameworks, databases, cloud platforms,
  ML/AI techniques, DevOps tooling, developer tools, and named engineering concepts
  (e.g. "System Design", "Distributed Systems").
- Skip soft skills, degrees, years of experience, company perks and the company's own products.
- importance = "required" when the posting presents it as a must-have; "preferred" for
  nice-to-haves ("bonus", "plus", "familiarity with").
- Use the widely used canonical name ("PostgreSQL", not "Postgres DB").
- Skills mentioned only in the company description, not as part of this role, do not count.
- If the posting is not a technical role, return an empty list."""


def build_prompt(job: Job) -> str:
    return (
        f"Title: {job.title}\nCompany: {job.company}\n\n"
        f"Description:\n{job.description[:MAX_DESCRIPTION_CHARS]}"
    )


class SkillResolver:
    """Maps raw skill names to canonical skill ids, creating skills as needed."""

    def __init__(self, session: Session, embedder: Embedder):
        self.session = session
        self.embedder = embedder

    def resolve(self, name: str, category: Category) -> int:
        name = _VERSION_SUFFIX.sub("", name.strip()) or name.strip()  # "Python 3.11" -> "Python"
        alias = " ".join(name.lower().split())
        found = self.session.scalar(select(SkillAlias.skill_id).where(SkillAlias.alias == alias))
        if found is not None:
            return found

        canonical = SEED_ALIASES.get(alias, name.strip())
        vector = self.embedder.embed_documents([canonical])[0]
        skill_id = self.session.scalar(select(Skill.id).where(Skill.name == canonical))
        if skill_id is None:
            row = self.session.execute(
                text(
                    "SELECT id, 1 - (embedding <=> CAST(:v AS vector)) AS sim FROM skills "
                    "WHERE category = :c ORDER BY embedding <=> CAST(:v AS vector) LIMIT 1"
                ),
                {"v": str(vector), "c": category},
            ).first()
            if row is not None and row.sim >= MERGE_THRESHOLD:
                skill_id = row.id
            else:
                skill_id = self.session.execute(
                    pg_insert(Skill)
                    .values(name=canonical, category=category, embedding=vector)
                    .returning(Skill.id)
                ).scalar_one()
        self.session.execute(
            pg_insert(SkillAlias).values(alias=alias, skill_id=skill_id).on_conflict_do_nothing()
        )
        return skill_id


def extract_job_skills(
    session: Session, llm: StructuredLLM, embedder: Embedder, job_ids: Iterable[int]
) -> int:
    """Extract skills for the given jobs, skipping any whose current content is already done.

    Returns the number of jobs sent to the LLM."""
    jobs = session.scalars(select(Job).where(Job.id.in_(list(job_ids)))).all()
    done = dict(
        session.execute(
            select(JobSkillExtraction.job_id, JobSkillExtraction.content_hash).where(
                JobSkillExtraction.job_id.in_([j.id for j in jobs])
            )
        ).all()
    )
    resolver = SkillResolver(session, embedder)
    extracted = 0
    for job in jobs:
        if done.get(job.id) == job.content_hash:
            continue
        result = llm.extract(SYSTEM_PROMPT, build_prompt(job), ExtractedSkills)

        # A posting may name one skill twice (once required, once "a plus"): keep the stronger.
        importance: dict[int, str] = {}
        for item in result.skills:
            skill_id = resolver.resolve(item.name, item.category)
            if importance.get(skill_id) != "required":
                importance[skill_id] = item.importance

        session.query(JobSkill).filter(JobSkill.job_id == job.id).delete()
        if importance:
            session.execute(
                pg_insert(JobSkill).values(
                    [{"job_id": job.id, "skill_id": s, "importance": i} for s, i in importance.items()]
                )
            )
        stmt = pg_insert(JobSkillExtraction).values(
            job_id=job.id, content_hash=job.content_hash, model=llm.model_name
        )
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[JobSkillExtraction.job_id],
                set_={
                    "content_hash": stmt.excluded.content_hash,
                    "model": stmt.excluded.model,
                    "extracted_at": text("now()"),
                },
            )
        )
        extracted += 1
    return extracted
