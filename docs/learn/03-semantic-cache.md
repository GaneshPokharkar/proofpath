# 3. Semantic caching: answer "the same question" without calling the model

## The problem
The skills report for a role (what employers ask for) costs ~30 AI calls the first time. Students
type the same role many ways: "Backend Engineer", "back-end developer", "backend dev". An exact
text cache misses all of those.

## What we measured first (`python -m evals.semantic_cache`)
40 labelled pairs of role names, 20 that mean the same role and 20 that don't:

| Rule | Same-role pairs reused | Wrong reuses | Precision |
|---|---|---|---|
| A. embedding similarity ≥ 0.85 | 17/20 | 8/20 | 0.68 |
| B. embedding similarity ≥ 0.93 | 12/20 | 0/20 | 1.00 |
| **C. canonical spelling, then ≥ 0.93 with the same seniority words** | **17/20** | **0/20** | **1.00** |

Embeddings alone were unsafe: "java developer" vs "javascript developer" scores 0.906, higher than
the true synonyms "ml engineer" vs "machine learning engineer" (0.891). A wrong hit (giving a Java
student a JavaScript report) is worse than a miss (one extra AI call), so precision comes first.

## The design (`app/market/demand.py`)
1. **Canonicalise** (`canonical_role`): fix spellings and abbreviations ("front end" → frontend,
   "ML" → machine learning, "SDE" → software development engineer, "dev" → developer). These now
   hit the exact cache.
2. **Exact lookup** on the canonical role plus scope (location, remote, India, sample size).
3. **Semantic lookup**: embed the role, take the nearest cached role with the same scope using
   pgvector (`role_vec <=> :v`), accept only if similarity ≥ 0.93 **and** the seniority words
   (intern, senior, lead…) match.
4. Every hit is traced as `cache` or `semantic-cache`, so `/admin` shows the real hit rate.

## Interview questions
**How do you choose a similarity threshold?** Label pairs from real traffic, sweep the
threshold, plot precision and recall, and choose based on the cost of a false hit vs a miss.
Don't guess 0.8 because a blog said so.

**Why can embeddings confuse "java" and "javascript"?** Embeddings capture topic and context;
both are programming-language developer roles, so they sit close together. Embedding similarity
is not equivalence, which is why exact rules and guards (seniority words) sit in front.

**When should you NOT use a semantic cache?** When answers depend on user-specific context (a
student's resume), on fast-changing data, or when a wrong answer is costly. ProofPath caches role
reports (shared, public data) but never chat replies.

**How does it invalidate?** Reports expire after 7 days, and the scope in the key keeps different
filters (India-only, remote) apart.
