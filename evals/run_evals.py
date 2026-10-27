"""Run the 100-case eval set against real models and the live ERP bridge.

    python -m evals.run_evals                       # default routing: cheap classify, strong reasoning
    python -m evals.run_evals --cheap claude-opus-5-5 --label strong-only
    python -m evals.run_evals --only K01,K02 --concurrency 2

Needs: ANTHROPIC_API_KEY, the ERP bridge (docker compose up) with the seeded Odoo.
Proposals made by action cases are rejected straight after scoring, so the
eval never writes to the ERP.

Writes evals/results/<label>.json, evals/results/<label>.md and, for the
default label, evals/results/badge.json (shields.io endpoint for the README).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
from evals.scoring import load_cases, score
from langgraph.checkpoint.memory import InMemorySaver

from ops_copilot.config import get_settings
from ops_copilot.graph import AgentDeps, build_graph
from ops_copilot.llm import AnthropicLLM
from ops_copilot.rag.embeddings import make_embedder
from ops_copilot.rag.ingest import ingest
from ops_copilot.rag.retriever import Retriever, make_reranker
from ops_copilot.rag.store import MemoryStore
from ops_copilot.service import CopilotService, MemoryRunLog, User
from ops_copilot.tools import BridgeAuth, McpErpTools

RESULTS = Path(__file__).parent / "results"


async def bridge_token(bridge: str, client_id: str) -> str:
    async with httpx.AsyncClient(trust_env=False) as http:
        resp = await http.post(
            f"{bridge}/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": f"{client_id}-demo-secret",
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


async def reject(bridge: str, approver_token: str, action_id: str) -> None:
    async with httpx.AsyncClient(trust_env=False) as http:
        await http.post(
            f"{bridge}/api/actions/{action_id}/reject",
            json={"note": "eval run: auto-rejected"},
            headers={"Authorization": f"Bearer {approver_token}"},
        )


async def build_service(cheap: str, strong: str) -> CopilotService:
    s = get_settings()
    embedder = make_embedder(s.embedder, s.embedding_model)
    store = MemoryStore()
    await ingest(store, embedder, s.corpus_dir)
    retriever = Retriever(store, embedder, make_reranker(s.reranker, s.rerank_model, s.cohere_api_key))
    erp = McpErpTools(s.bridge_url, BridgeAuth(s.bridge_url, s.bridge_client_id, s.bridge_client_secret))
    deps = AgentDeps(
        llm=AnthropicLLM(cheap, strong, s.strong_effort),
        retriever=retriever,
        erp=erp,
        today=s.business_date,
        callback_base=None,
        max_steps=s.max_agent_steps,
        top_k=s.retrieve_top_k,
    )
    return CopilotService(build_graph(deps, InMemorySaver()), MemoryRunLog())


async def run(args: argparse.Namespace) -> dict:
    s = get_settings()
    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    service = await build_service(args.cheap, args.strong)
    tokens = {"store_manager": await bridge_token(s.bridge_url, "store-klcc")}
    approver = await bridge_token(s.bridge_url, "approver-ops")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(case: dict) -> dict:
        async with sem:
            role = case.get("role", "staff")
            user = User(id=f"eval-{case['id']}", role=role, erp_token=tokens.get(role) if case.get("scoped") else None)
            started = time.perf_counter()
            result = (await service.ask(case["question"], user)).model_dump()
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            if result.get("action") and result["action"].get("status") == "pending":
                await reject(s.bridge_url, approver, result["action"]["id"])
            graded = score(case, result)
            mark = "PASS" if graded["passed"] else "FAIL " + ",".join(k for k, v in graded["checks"].items() if not v)
            print(f"{case['id']:>5} {mark:<28} ${result['cost_usd']:.4f} {result['latency_ms']}ms")
            return {"case": case, "result": result, "score": graded}

    rows = await asyncio.gather(*(one(c) for c in cases))
    passed = sum(r["score"]["passed"] for r in rows)
    costs = [r["result"]["cost_usd"] for r in rows]
    latencies = sorted(r["result"]["latency_ms"] for r in rows)
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["case"]["category"], []).append(r["score"]["passed"])
    summary = {
        "label": args.label,
        "cheap_model": args.cheap,
        "strong_model": args.strong,
        "cases": len(rows),
        "passed": passed,
        "score": round(passed / len(rows), 4) if rows else 0.0,
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "total_cost_usd": round(sum(costs), 4),
        "mean_cost_per_run_usd": round(statistics.mean(costs), 5),
        "p50_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(len(latencies) * 0.95) - 1],
        "embedder": s.embedder,
        "reranker": s.reranker,
    }
    return {"summary": summary, "rows": rows}


def write(report: dict, label: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{label}.json").write_text(json.dumps(report, indent=2, default=str))
    s = report["summary"]
    lines = [
        f"# Eval results: {label}",
        "",
        f"**{s['passed']}/{s['cases']} passed** ({s['score']:.0%})",
        "",
        f"- Models: classify `{s['cheap_model']}`, reason `{s['strong_model']}`",
        f"- Cost: ${s['total_cost_usd']} total, ${s['mean_cost_per_run_usd']} per run",
        f"- Latency: p50 {s['p50_latency_ms']} ms, p95 {s['p95_latency_ms']} ms",
        "",
        "| Category | Passed |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in s["by_category"].items()],
        "",
        "## Failures",
        "",
    ]
    for r in report["rows"]:
        if not r["score"]["passed"]:
            failed = ", ".join(k for k, v in r["score"]["checks"].items() if not v)
            lines.append(f"- **{r['case']['id']}** ({failed}): {r['case']['question']}")
    (RESULTS / f"{label}.md").write_text("\n".join(lines) + "\n")
    if label == "default":
        color = "brightgreen" if s["score"] >= 0.9 else "yellow" if s["score"] >= 0.75 else "red"
        (RESULTS / "badge.json").write_text(
            json.dumps({"schemaVersion": 1, "label": "evals", "message": f"{s['passed']}/{s['cases']}", "color": color})
        )


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--cheap", default=s.cheap_model)
    p.add_argument("--strong", default=s.strong_model)
    p.add_argument("--label", default="default")
    p.add_argument("--only")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    report = asyncio.run(run(args))
    write(report, args.label)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
