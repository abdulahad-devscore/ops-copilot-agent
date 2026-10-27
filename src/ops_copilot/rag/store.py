"""Chunk stores with dense (vector) and sparse (keyword) search.

pgvector is the default: one database, one backup, one permission model.
Keyword search uses Postgres full-text search, because SKUs, invoice numbers
and supplier codes ("TH-SA-6104", "SUP-KVP") are exactly what pure vector
search gets wrong. `MemoryStore` implements the same interface with numpy
and BM25 for tests and offline runs.

Both stores filter by role *inside* the query, so a chunk the user may not
see is never even a candidate.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

import numpy as np

from ops_copilot.rag.documents import Chunk
from ops_copilot.rag.embeddings import tokenize


class ChunkStore(Protocol):
    async def setup(self, dim: int) -> None: ...
    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int: ...
    async def prune(self, keep_ids: set[str]) -> int: ...
    async def dense(self, qvec: np.ndarray, k: int, roles: list[str]) -> list[tuple[Chunk, float]]: ...
    async def sparse(self, query: str, k: int, roles: list[str]) -> list[tuple[Chunk, float]]: ...
    async def count(self) -> int: ...


class MemoryStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self._bm25: tuple[dict[str, Counter], dict[str, int], float] | None = None

    async def setup(self, dim: int) -> None:
        pass

    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        for c, v in zip(chunks, vectors, strict=True):
            self.chunks[c.id] = c
            self.vectors[c.id] = v
        self._bm25 = None
        return len(chunks)

    async def prune(self, keep_ids: set[str]) -> int:
        gone = [i for i in self.chunks if i not in keep_ids]
        for i in gone:
            del self.chunks[i], self.vectors[i]
        self._bm25 = None
        return len(gone)

    async def count(self) -> int:
        return len(self.chunks)

    def _visible(self, roles: list[str]) -> list[Chunk]:
        return [c for c in self.chunks.values() if set(c.access) & set(roles)]

    async def dense(self, qvec: np.ndarray, k: int, roles: list[str]) -> list[tuple[Chunk, float]]:
        visible = self._visible(roles)
        if not visible:
            return []
        mat = np.stack([self.vectors[c.id] for c in visible])
        scores = mat @ qvec / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(qvec) or 1.0) + 1e-9)
        order = np.argsort(-scores)[:k]
        return [(visible[i], float(scores[i])) for i in order]

    def _index(self) -> tuple[dict[str, Counter], dict[str, int], float]:
        if self._bm25 is None:
            tfs = {cid: Counter(tokenize(c.embed_text())) for cid, c in self.chunks.items()}
            df: Counter = Counter()
            for tf in tfs.values():
                df.update(tf.keys())
            avg = sum(sum(tf.values()) for tf in tfs.values()) / max(1, len(tfs))
            self._bm25 = (tfs, dict(df), avg)
        return self._bm25

    async def sparse(self, query: str, k: int, roles: list[str]) -> list[tuple[Chunk, float]]:
        tfs, df, avg = self._index()
        n = len(tfs)
        terms = set(tokenize(query))
        scored = []
        for c in self._visible(roles):
            tf = tfs[c.id]
            length = sum(tf.values())
            s = 0.0
            for t in terms:
                if t in tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * length / avg))
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


class PgVectorStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            from pgvector.psycopg import register_vector_async
            from psycopg_pool import AsyncConnectionPool

            async def configure(conn):
                await register_vector_async(conn)

            self._pool = AsyncConnectionPool(self.dsn, min_size=1, max_size=5, configure=configure, open=False)
            await self._pool.open()
        return self._pool

    async def setup(self, dim: int) -> None:
        import psycopg

        # The extension must exist before connections register the vector type.
        async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id text PRIMARY KEY,
                    doc_id text NOT NULL,
                    title text NOT NULL,
                    doc_type text,
                    section text,
                    text text NOT NULL,
                    access text[] NOT NULL,
                    effective text,
                    content_hash text,
                    embedding vector({dim}) NOT NULL,
                    tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', title || ' ' || coalesce(section, '') || ' ' || text)) STORED
                )""")
            await conn.execute("CREATE INDEX IF NOT EXISTS rag_chunks_tsv ON rag_chunks USING gin (tsv)")
            await conn.execute("CREATE INDEX IF NOT EXISTS rag_chunks_access ON rag_chunks USING gin (access)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops)"
            )

    async def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                """INSERT INTO rag_chunks (id, doc_id, title, doc_type, section, text, access, effective,
                                           content_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, title = EXCLUDED.title,
                       doc_type = EXCLUDED.doc_type, section = EXCLUDED.section, text = EXCLUDED.text,
                       access = EXCLUDED.access, effective = EXCLUDED.effective,
                       content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding
                   WHERE rag_chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                      OR rag_chunks.access IS DISTINCT FROM EXCLUDED.access""",
                [
                    (
                        c.id,
                        c.doc_id,
                        c.title,
                        c.doc_type,
                        c.section,
                        c.text,
                        c.access,
                        c.effective,
                        c.content_hash,
                        v.astype(np.float32),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        return len(chunks)

    async def prune(self, keep_ids: set[str]) -> int:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("DELETE FROM rag_chunks WHERE NOT (id = ANY(%s))", (list(keep_ids),))
            return cur.rowcount

    async def count(self) -> int:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            row = await (await conn.execute("SELECT count(*) FROM rag_chunks")).fetchone()
            return row[0]

    _COLS = "id, doc_id, title, doc_type, section, text, access, effective, content_hash"

    @staticmethod
    def _row(r) -> Chunk:
        return Chunk(
            id=r[0],
            doc_id=r[1],
            title=r[2],
            doc_type=r[3],
            section=r[4],
            text=r[5],
            access=list(r[6]),
            effective=r[7],
            content_hash=r[8],
        )

    async def dense(self, qvec: np.ndarray, k: int, roles: list[str]) -> list[tuple[Chunk, float]]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    f"""SELECT {self._COLS}, 1 - (embedding <=> %s) AS score FROM rag_chunks
                    WHERE access && %s ORDER BY embedding <=> %s LIMIT %s""",
                    (qvec.astype(np.float32), roles, qvec.astype(np.float32), k),
                )
            ).fetchall()
        return [(self._row(r), float(r[9])) for r in rows]

    async def sparse(self, query: str, k: int, roles: list[str]) -> list[tuple[Chunk, float]]:
        terms = sorted({t for t in tokenize(query) if "-" not in t and len(t) > 1})
        if not terms:
            return []
        pool = await self._get_pool()
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    f"""SELECT {self._COLS}, ts_rank_cd(tsv, q) AS score
                    FROM rag_chunks, to_tsquery('english', %s) q
                    WHERE tsv @@ q AND access && %s ORDER BY score DESC LIMIT %s""",
                    (" | ".join(terms), roles, k),
                )
            ).fetchall()
        return [(self._row(r), float(r[9])) for r in rows]

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
