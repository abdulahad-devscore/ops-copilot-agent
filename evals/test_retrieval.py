"""Deterministic retrieval evals: no LLM, no API key, run on every pull request.

For every case with `docs`, the right document must be in the top 5 for the
case's role (hit@5), and we report MRR. A second check proves permission-aware
retrieval: across every question, a role never receives a chunk it may not read.

    EVAL_EMBEDDER=fastembed pytest evals/test_retrieval.py -s    # what CI runs
"""

import json
import os
from pathlib import Path

import pytest
from evals.scoring import load_cases

from ops_copilot.rag.embeddings import make_embedder
from ops_copilot.rag.ingest import ingest
from ops_copilot.rag.retriever import ROLE_ACCESS, Retriever, make_reranker
from ops_copilot.rag.store import MemoryStore

EMBEDDER = os.environ.get("EVAL_EMBEDDER", "hashing")
RERANKER = os.environ.get("EVAL_RERANKER", "none" if EMBEDDER == "hashing" else "fastembed")
# The hashing embedder is a keyword stand-in for offline runs, so it gets a lower bar.
MIN_HIT_AT_5 = float(os.environ.get("EVAL_MIN_HIT5", "0.85" if EMBEDDER == "fastembed" else "0.70"))
RESULTS = Path(__file__).parent / "results"


@pytest.fixture(scope="module")
async def retriever():
    store = MemoryStore()
    embedder = make_embedder(EMBEDDER, "BAAI/bge-small-en-v1.5")
    await ingest(store, embedder, "corpus")
    return Retriever(store, embedder, make_reranker(RERANKER, "Xenova/ms-marco-MiniLM-L-6-v2"))


async def test_hit_at_5(retriever):
    cases = [c for c in load_cases() if c.get("docs")]
    hits, rr, misses = 0, 0.0, []
    for case in cases:
        results = await retriever.search(case["question"], case.get("role", "staff"), top_k=5)
        ranked = [r.chunk.doc_id for r in results]
        found = next((i for i, d in enumerate(ranked) if d in case["docs"]), None)
        if found is None:
            misses.append({"id": case["id"], "question": case["question"], "expected": case["docs"], "got": ranked})
        else:
            hits += 1
            rr += 1 / (found + 1)
    hit5, mrr = hits / len(cases), rr / len(cases)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"retrieval-{EMBEDDER}.json").write_text(
        json.dumps(
            {
                "embedder": EMBEDDER,
                "reranker": RERANKER,
                "cases": len(cases),
                "hit_at_5": round(hit5, 4),
                "mrr": round(mrr, 4),
                "misses": misses,
            },
            indent=2,
        )
    )
    print(f"\nretrieval ({EMBEDDER} + {RERANKER}): hit@5 {hit5:.2%}, MRR {mrr:.3f} over {len(cases)} cases")
    for m in misses:
        print(f"  miss {m['id']}: {m['question']} -> {m['got']}")
    assert hit5 >= MIN_HIT_AT_5


async def test_no_permission_leaks(retriever):
    restricted = ["staff", "store_manager", "finance"]  # roles that must not see some documents
    for case in load_cases():
        for role in restricted:
            for r in await retriever.search(case["question"], role, top_k=8):
                visible = set(r.chunk.access) & set(ROLE_ACCESS[role])
                assert visible, f"{role} received {r.chunk.doc_id} (access {r.chunk.access}) for {case['id']}"
