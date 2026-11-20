"""Fetch boards of each source type and fail if a type no longer parses at all (its API changed)."""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ingest.registry import load_sources  # noqa: E402

ok_types: set[str] = set()
tried: dict[str, list[str]] = {}
with httpx.Client(timeout=60, headers={"User-Agent": "ProofPath source check"}) as client:
    for src in load_sources(Path(__file__).resolve().parent.parent / "sources.yaml"):
        kind = src.key.split(":")[0]
        if kind in ok_types or kind == "remotive" or len(tried.get(kind, [])) >= 3:
            continue  # Remotive allows only a few fetches a day; 3 tries per type is enough
        try:
            jobs = src.fetch(client)
            assert jobs, "no jobs"
            assert all(j.title and j.url and j.external_id for j in jobs[:20]), "missing fields"
            ok_types.add(kind)
            print(f"ok    {src.key}: {len(jobs)} jobs")
        except Exception as exc:  # noqa: BLE001
            tried.setdefault(kind, []).append(f"{src.key}: {exc}")
            print(f"warn  {src.key}: {exc}")
broken = sorted(set(tried) - ok_types)
for kind in broken:
    print(f"FAIL  every {kind} board failed")
sys.exit(1 if broken else 0)
