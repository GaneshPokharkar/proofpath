"""ORM models for students, repos and evidence (migration 0003)."""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    github_login: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    student_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    full_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    is_fork: Mapped[bool] = mapped_column(Boolean, nullable=False)
    stars: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    pushed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    languages: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RepoEvidence(Base):
    __tablename__ = "repo_evidence"

    repo_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    skill_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True
    )
    source: Mapped[str] = mapped_column(String(16), primary_key=True)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
