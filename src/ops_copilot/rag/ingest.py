"""Ingest the corpus: load -> chunk -> embed -> upsert, and prune chunks whose
documents were removed. Safe to re-run; unchanged chunks are not rewritten.

    python -m ops_copilot.rag.ingest
"""

from __future__ import annotations

import asyncio
import logging
import time

from ops_copilot.config import get_settings
from ops_copilot.rag.documents import Chunk, chunk_document, load_corpus
from ops_copilot.rag.embeddings import Embedder, make_embedder
from ops_copilot.rag.store import ChunkStore, MemoryStore, PgVectorStore

log = logging.getLogger(__name__)


async def ingest(store: ChunkStore, embedder: Embedder, corpus_dir: str) -> dict:
    started = time.perf_counter()
    docs = load_corpus(corpus_dir)
    chunks: list[Chunk] = [c for d in docs for c in chunk_document(d)]
    await store.setup(embedder.dim)
    vectors = embedder.embed_documents([c.embed_text() for c in chunks])
    await store.upsert(chunks, vectors)
    pruned = await store.prune({c.id for c in chunks})
    stats = {
        "documents": len(docs),
        "chunks": len(chunks),
        "pruned": pruned,
        "embedder": embedder.name,
        "seconds": round(time.perf_counter() - started, 1),
    }
    log.info("ingested %s", stats)
    return stats


def make_store(kind: str, dsn: str) -> ChunkStore:
    return PgVectorStore(dsn) if kind == "pgvector" else MemoryStore()


async def main() -> None:
    s = get_settings()
    store = make_store(s.vector_store, s.database_url)
    print(await ingest(store, make_embedder(s.embedder, s.embedding_model), s.corpus_dir))
    if isinstance(store, PgVectorStore):
        await store.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
