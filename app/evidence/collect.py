"""Store a student's repos and evidence, and compare evidence with what the market asks for."""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.embeddings import Embedder
from app.evidence.detect import detect
from app.evidence.github import GitHubClient
from app.market.demand import MarketRead
from app.market.skills import SkillResolver
from app.models_evidence import Repo, RepoEvidence, Student


@dataclass(frozen=True)
class AnalysisSummary:
    login: str
    repos: int
    forks_skipped: int
    evidence: int


def analyze_student(
    session: Session, github: GitHubClient, embedder: Embedder, login: str, repo_limit: int = 15
) -> AnalysisSummary:
    student_id = session.execute(
        pg_insert(Student)
        .values(github_login=login.lower())
        .on_conflict_do_update(index_elements=[Student.github_login], set_={"github_login": login.lower()})
        .returning(Student.id)
    ).scalar_one()
    resolver = SkillResolver(session, embedder)
    repos = forks = rows = 0
    for raw in github.list_repos(login, limit=repo_limit):
        snap = github.snapshot(raw)
        values = dict(
            student_id=student_id,
            full_name=snap.full_name,
            description=snap.description,
            url=snap.url,
            is_fork=snap.is_fork,
            stars=snap.stars,
            pushed_at=snap.pushed_at,
            languages=snap.languages,
            analyzed_at=datetime.now(UTC),
        )
        stmt = pg_insert(Repo).values(**values)
        repo_id = session.execute(
            stmt.on_conflict_do_update(index_elements=[Repo.full_name], set_=values).returning(Repo.id)
        ).scalar_one()
        # Re-analysis replaces old evidence: a removed dependency is no longer proof.
        session.query(RepoEvidence).filter(RepoEvidence.repo_id == repo_id).delete()
        repos += 1
        if snap.is_fork:
            forks += 1
            continue
        found = detect(snap)
        if found:
            session.execute(
                pg_insert(RepoEvidence)
                .values(
                    [
                        {
                            "repo_id": repo_id,
                            "skill_id": resolver.resolve(e.skill, e.category),
                            "source": e.source,
                            "detail": e.detail,
                        }
                        for e in found
                    ]
                )
                .on_conflict_do_nothing()
            )
            rows += len(found)
    return AnalysisSummary(login=login, repos=repos, forks_skipped=forks, evidence=rows)


@dataclass(frozen=True)
class SkillCoverage:
    skill: str
    demand_share: float  # share of sampled postings asking for it
    required: int
    proven: bool
    repos: list[str]  # repos with evidence
    details: list[str]


@dataclass(frozen=True)
class Readiness:
    role: str
    score: float  # demand-weighted share of skills with evidence, 0..1
    skills: list[SkillCoverage]


def readiness(session: Session, login: str, market: MarketRead) -> Readiness:
    """For each skill the market asks for, does this student's GitHub show evidence of it?
    The score weights each skill by how many postings want it, so missing Python matters more
    than missing a niche library."""
    names = [s.skill for s in market.skills]
    rows = session.execute(
        text(
            """
            SELECT sk.name, r.full_name, e.detail
            FROM repo_evidence e
            JOIN repos r ON r.id = e.repo_id
            JOIN students st ON st.id = r.student_id
            JOIN skills sk ON sk.id = e.skill_id
            WHERE st.github_login = :login AND sk.name = ANY(:names)
            ORDER BY r.pushed_at DESC NULLS LAST
            """
        ),
        {"login": login.lower(), "names": names},
    ).all()
    proof: dict[str, tuple[list[str], list[str]]] = {}
    for name, repo, detail in rows:
        repos, details = proof.setdefault(name, ([], []))
        if repo not in repos:
            repos.append(repo)
        details.append(f"{repo}: {detail}")

    coverage = [
        SkillCoverage(
            skill=s.skill,
            demand_share=s.share,
            required=s.required,
            proven=s.skill in proof,
            repos=proof.get(s.skill, ([], []))[0],
            details=proof.get(s.skill, ([], []))[1][:3],
        )
        for s in market.skills
    ]
    total = sum(c.demand_share for c in coverage)
    score = sum(c.demand_share for c in coverage if c.proven) / total if total else 0.0
    return Readiness(role=market.role, score=round(score, 3), skills=coverage)
