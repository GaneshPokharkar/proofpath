"""Measure the semantic cache's decision rule on labelled role pairs.

    python -m evals.semantic_cache

A false hit (two different roles sharing one skills report) is worse than a miss (one extra AI
call), so the rule is tuned for precision first. Compares three rules:
  A. embedding similarity >= 0.85
  B. embedding similarity >= 0.93
  C. the shipped rule: canonical spelling first, then similarity >= 0.93 with equal seniority words
"""

import numpy as np

from app.config import get_settings
from app.embeddings import FastEmbedEmbedder
from app.market.demand import SEMANTIC_MIN, _levels, canonical_role

SAME = [
    ("backend engineer", "backend developer"),
    ("data analyst", "data analytics"),
    ("ml engineer", "machine learning engineer"),
    ("frontend developer", "front end developer"),
    ("react developer", "reactjs developer"),
    ("data scientist", "data science"),
    ("devops engineer", "dev ops engineer"),
    ("software engineer intern", "software engineering intern"),
    ("sde", "software development engineer"),
    ("python developer", "python dev"),
    ("full stack developer", "fullstack developer"),
    ("ai engineer", "artificial intelligence engineer"),
    ("node.js developer", "nodejs developer"),
    ("swe intern", "software engineer intern"),
    ("back-end developer", "backend developer"),
    ("ui ux designer", "ui/ux designer"),
    ("cloud engineer", "cloud engineering"),
    ("business analyst", "business analyst role"),
    ("android developer", "android app developer"),
    ("data engineer", "data engineering"),
]
DIFFERENT = [
    ("backend engineer", "frontend engineer"),
    ("data analyst", "data scientist"),
    ("ml engineer", "data engineer"),
    ("software engineer", "software engineer intern"),
    ("java developer", "javascript developer"),
    ("android developer", "ios developer"),
    ("devops engineer", "security engineer"),
    ("product manager", "project manager"),
    ("data engineer", "data analyst"),
    ("qa engineer", "qa intern"),
    ("senior data scientist", "data scientist"),
    ("react developer", "react native developer"),
    ("frontend developer", "fullstack developer"),
    ("machine learning intern", "machine learning engineer"),
    ("cloud engineer", "network engineer"),
    ("business analyst", "data analyst"),
    ("c++ developer", "c# developer"),
    ("sales engineer", "software engineer"),
    ("ui designer", "graphic designer"),
    ("lead engineer", "engineer"),
]


def main() -> None:
    e = FastEmbedEmbedder(get_settings().embedding_model)

    def sim(a: str, b: str) -> float:
        v = np.array([e.embed_query(a), e.embed_query(b)])
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return float(v[0] @ v[1])

    rules = {
        "A embeddings >= 0.85": lambda a, b: sim(a, b) >= 0.85,
        "B embeddings >= 0.93": lambda a, b: sim(a, b) >= 0.93,
        "C canonical + 0.93 + levels": lambda a, b: (
            canonical_role(a) == canonical_role(b)
            or (
                sim(canonical_role(a), canonical_role(b)) >= SEMANTIC_MIN
                and _levels(canonical_role(a)) == _levels(canonical_role(b))
            )
        ),
    }
    for name, rule in rules.items():
        tp = sum(rule(a, b) for a, b in SAME)
        fp = sum(rule(a, b) for a, b in DIFFERENT)
        precision = tp / (tp + fp) if tp + fp else 1.0
        print(
            f"{name:<30} hits on same roles {tp}/{len(SAME)} (recall {tp / len(SAME):.2f}), "
            f"false hits {fp}/{len(DIFFERENT)}, precision {precision:.2f}"
        )
        if name.startswith("C"):
            for a, b in DIFFERENT:
                if rule(a, b):
                    print(f"   false hit: {a!r} ~ {b!r}")


if __name__ == "__main__":
    main()
