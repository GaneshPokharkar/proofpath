"""Command line: `python -m app.cli ingest` and `python -m app.cli search "query"`."""

import argparse
import logging
import sys
import time
from datetime import timedelta

import httpx

from app.config import get_settings
from app.db import session_scope
from app.embeddings import FastEmbedEmbedder
from app.ingest.pipeline import run_ingestion
from app.ingest.registry import load_sources
from app.search import SearchFilters, hybrid_search


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    sources = load_sources(settings.sources_file)
    if args.only:
        wanted = set(args.only)
        sources = [s for s in sources if s.key in wanted or s.key.split(":")[0] in wanted]
        if not sources:
            print(f"no configured source matches {sorted(wanted)}", file=sys.stderr)
            return 2

    embedder = FastEmbedEmbedder(settings.embedding_model)
    headers = {"User-Agent": settings.user_agent}
    start = time.perf_counter()
    with httpx.Client(timeout=settings.http_timeout_s, headers=headers, follow_redirects=True) as client:
        report = run_ingestion(
            session_scope, sources, client, embedder, timedelta(days=settings.expire_after_days)
        )

    print(f"{'source':<24}{'fetched':>8}{'new':>6}{'changed':>8}{'same':>6}{'retired':>8}{'embedded':>9}")
    for r in report.runs:
        print(
            f"{r.source:<24}{r.fetched:>8}{r.inserted:>6}{r.updated:>8}"
            f"{r.unchanged:>6}{r.deactivated:>8}{r.embedded:>9}"
        )
    for key, error in report.failures.items():
        print(f"FAILED {key}: {error}", file=sys.stderr)
    print(f"done in {time.perf_counter() - start:.1f}s")
    # Non-zero exit when any source failed, so the scheduler and CI notice.
    return 0 if report.ok else 1


def cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    filters = SearchFilters(remote=args.remote, location=args.location)
    with session_scope() as session:
        start = time.perf_counter()
        hits = hybrid_search(session, embedder, args.query, filters, limit=args.limit)
        took = (time.perf_counter() - start) * 1000
    for i, h in enumerate(hits, 1):
        where = h.location or ("Remote" if h.remote else "?")
        print(f"{i:>2}. {h.title} | {h.company} | {where} | kw#{h.keyword_rank} sem#{h.semantic_rank}")
    print(f"{len(hits)} results in {took:.0f} ms")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="jobpilot")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="fetch all configured sources into the database")
    ingest.add_argument(
        "--only", nargs="+", help="source keys or types, e.g. remotive greenhouse:groww lever"
    )
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="run a hybrid search from the terminal")
    search.add_argument("query")
    search.add_argument("--remote", action=argparse.BooleanOptionalAction, default=None)
    search.add_argument("--location")
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
