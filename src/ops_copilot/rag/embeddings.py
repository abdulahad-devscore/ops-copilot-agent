"""Embedding models.

Default is a local ONNX model through fastembed (BAAI/bge-small-en-v1.5,
384 dimensions): no data leaves the machine, no per-call cost, and it is
baked into the Docker image so containers run offline. `HashingEmbedder` is a
deterministic bag-of-words stand-in for unit tests only; it is not a
semantic model and is never used for reported eval numbers.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # bge models expect an instruction prefix on queries; fastembed adds it in query_embed
        return np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)


_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    out = []
    for t in tokens:
        out.append(t)
        if "-" in t:  # TH-CW-2401 is also searchable as th, cw, 2401
            out.extend(t.split("-"))
    return out


class HashingEmbedder:
    """Feature-hashed unigrams + bigrams, L2-normalised. Tests only."""

    def __init__(self, dim: int = 384):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        for gram in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)]:
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norm = math.sqrt(float(v @ v)) or 1.0
        return v / norm

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def make_embedder(kind: str, model: str) -> Embedder:
    if kind == "fastembed":
        return FastEmbedEmbedder(model)
    if kind == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {kind}")
