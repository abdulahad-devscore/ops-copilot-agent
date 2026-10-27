"""Hybrid, permission-aware retrieval: dense + keyword, fused with reciprocal
rank fusion, then reranked by a cross-encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ops_copilot.rag.documents import Chunk
from ops_copilot.rag.embeddings import Embedder
from ops_copilot.rag.store import ChunkStore

# Which document audiences each role may read. A role sees its own documents
# and anything shared with all staff.
ROLE_ACCESS: dict[str, list[str]] = {
    "staff": ["staff"],
    "store_manager": ["staff", "store_manager"],
    "finance": ["staff", "finance"],
    "management": ["staff", "store_manager", "finance", "management"],
}


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None

    def as_source(self, label: str) -> dict:
        c = self.chunk
        return {
            "id": label,
            "chunk_id": c.id,
            "doc_id": c.doc_id,
            "title": c.title,
            "section": c.section,
            "text": c.text,
            "effective": c.effective,
        }


class Reranker(Protocol):
    name: str

    def scores(self, query: str, texts: list[str]) -> list[float]: ...


class NoReranker:
    name = "none"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(-i) for i in range(len(texts))]  # keep fused order


class FastEmbedReranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self._model = TextCrossEncoder(model)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return list(self._model.rerank(query, texts))


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        import cohere

        self.name = model
        self._client = cohere.ClientV2(api_key)
        self._model = model

    def scores(self, query: str, texts: list[str]) -> list[float]:
        resp = self._client.rerank(model=self._model, query=query, documents=texts)
        out = [0.0] * len(texts)
        for r in resp.results:
            out[r.index] = r.relevance_score
        return out


def make_reranker(kind: str, model: str, cohere_key: str | None = None) -> Reranker:
    if kind == "fastembed":
        return FastEmbedReranker(model)
    if kind == "cohere":
        if not cohere_key:
            raise ValueError("COHERE_API_KEY is required for the cohere reranker")
        return CohereReranker(cohere_key)
    return NoReranker()


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    def __init__(
        self, store: ChunkStore, embedder: Embedder, reranker: Reranker, candidates: int = 30, rerank_pool: int = 20
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_pool = rerank_pool

    async def search(self, query: str, role: str, top_k: int = 6) -> list[Retrieved]:
        roles = ROLE_ACCESS.get(role)
        if roles is None:
            raise ValueError(f"unknown role {role!r}")
        dense = await self.store.dense(self.embedder.embed_query(query), self.candidates, roles)
        sparse = await self.store.sparse(query, self.candidates, roles)
        by_id = {c.id: c for c, _ in dense + sparse}
        d_rank = {c.id: i for i, (c, _) in enumerate(dense)}
        s_rank = {c.id: i for i, (c, _) in enumerate(sparse)}
        fused = rrf([[c.id for c, _ in dense], [c.id for c, _ in sparse]])
        pool = sorted(fused, key=lambda cid: -fused[cid])[: self.rerank_pool]
        if not pool:
            return []
        texts = [by_id[cid].embed_text() for cid in pool]
        scores = self.reranker.scores(query, texts)
        ranked = sorted(zip(pool, scores, strict=True), key=lambda x: -x[1])
        out: list[Retrieved] = []
        per_doc: dict[str, int] = {}
        for cid, score in ranked:
            doc = by_id[cid].doc_id
            if per_doc.get(doc, 0) >= 2:  # diversity: at most 2 chunks per document
                continue
            per_doc[doc] = per_doc.get(doc, 0) + 1
            out.append(Retrieved(by_id[cid], float(score), d_rank.get(cid), s_rank.get(cid)))
            if len(out) == top_k:
                break
        return out
