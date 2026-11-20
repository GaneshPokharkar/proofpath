"""Text embeddings. The real model runs locally (fastembed, ONNX); tests use HashingEmbedder."""

import hashlib
import math
import os
import re
from typing import Protocol

from app.models import EMBEDDING_DIM


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 by default: 384 dimensions, runs on CPU, no API key."""

    def __init__(self, model_name: str, batch_size: int = 64):
        from fastembed import TextEmbedding  # imported lazily: loading the model takes seconds

        self.model_name = model_name
        # On read-only hosts (Vercel) only /tmp is writable: FASTEMBED_CACHE_PATH points there.
        self._model = TextEmbedding(model_name, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
        self._batch_size = batch_size
        probe = next(iter(self._model.embed(["dimension check"])))
        if len(probe) != EMBEDDING_DIM:
            raise ValueError(
                f"{model_name} produces {len(probe)}-d vectors but the schema expects {EMBEDDING_DIM}"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts, batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        # bge models retrieve best when the query (not the passage) carries an instruction prefix;
        # fastembed's query_embed adds the right one for the model.
        return next(iter(self._model.query_embed(text))).tolist()


_TOKEN = re.compile(r"[a-z0-9+#]+")


class HashingEmbedder:
    """Deterministic bag-of-words vectors for tests: texts sharing words get similar vectors."""

    model_name = "test-hashing"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        for token in _TOKEN.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBEDDING_DIM
            vec[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            # A zero vector has no cosine distance (pgvector returns NaN): give empty text a direction.
            vec[0], norm = 1.0, 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
