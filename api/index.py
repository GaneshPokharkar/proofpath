"""Vercel entry point: Vercel's Python runtime serves the ASGI `app` exported here."""

import os

# Vercel's filesystem is read-only except /tmp, so the embedding model is cached there
# (downloaded once per cold start, ~65 MB).
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed")

from app.api.main import app  # noqa: E402

__all__ = ["app"]
