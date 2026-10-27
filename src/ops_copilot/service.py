"""Runs and resumes agent threads, and records what each run cost."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Any, Protocol

from langgraph.types import Command
from pydantic import BaseModel

from ops_copilot.graph import run_cost
from ops_copilot.tracing import get_tracer


class User(BaseModel):
    id: str
    role: str = "staff"
    erp_token: str | None = None  # the user's own bridge token, if the caller has one


class RunResult(BaseModel):
    thread_id: str
    status: str  # completed | awaiting_approval | failed
    route: str | None = None
    answer: str | None = None
    citations: list[dict] = []
    action: dict | None = None
    tools_called: list[str] = []
    tool_log: list[dict] = []
    cost_usd: float = 0.0
    cost_by_model: dict[str, float] = {}
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    trace_url: str | None = None


class RunLog(Protocol):
    async def save(self, result: RunResult, question: str, user: User) -> None: ...
    async def get(self, thread_id: str) -> dict | None: ...
    async def recent(self, limit: int) -> list[dict]: ...


class MemoryRunLog:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    async def save(self, result: RunResult, question: str, user: User) -> None:
        self.rows[result.thread_id] = {"question": question, "user_id": user.id, **result.model_dump()}

    async def get(self, thread_id: str) -> dict | None:
        return self.rows.get(thread_id)

    async def recent(self, limit: int) -> list[dict]:
        return list(self.rows.values())[-limit:][::-1]


class PgRunLog:
    def __init__(self, pool):
        self.pool = pool

    async def setup(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS copilot_runs (
                    thread_id text PRIMARY KEY, question text, user_id text, status text, route text,
                    answer text, result jsonb, cost_usd numeric(12, 6), created_at timestamptz DEFAULT now(),
                    updated_at timestamptz DEFAULT now())""")

    async def save(self, result: RunResult, question: str, user: User) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO copilot_runs (thread_id, question, user_id, status, route, answer, result, cost_usd)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (thread_id) DO UPDATE SET status = EXCLUDED.status, answer = EXCLUDED.answer,
                     result = EXCLUDED.result, cost_usd = EXCLUDED.cost_usd, updated_at = now()""",
                (
                    result.thread_id,
                    question,
                    user.id,
                    result.status,
                    result.route,
                    result.answer,
                    json.dumps(result.model_dump()),
                    result.cost_usd,
                ),
            )

    async def get(self, thread_id: str) -> dict | None:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT question, user_id, result FROM copilot_runs WHERE thread_id = %s", (thread_id,)
                )
            ).fetchone()
        return {"question": row[0], "user_id": row[1], **row[2]} if row else None

    async def recent(self, limit: int) -> list[dict]:
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT question, user_id, result FROM copilot_runs ORDER BY updated_at DESC LIMIT %s", (limit,)
                )
            ).fetchall()
        return [{"question": r[0], "user_id": r[1], **r[2]} for r in rows]


class CopilotService:
    def __init__(self, graph, runs: RunLog):
        self.graph = graph
        self.runs = runs
        self.tracer = get_tracer()
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def ask(self, question: str, user: User, thread_id: str | None = None) -> RunResult:
        thread_id = thread_id or str(uuid.uuid4())
        state = {"question": question, "user": user.model_dump(), "thread_id": thread_id, "usage": [], "tool_log": []}
        return await self._drive(thread_id, state, question, user)

    async def resume(self, thread_id: str, payload: dict | None = None) -> RunResult:
        snapshot = await self.graph.aget_state(self._config(thread_id))
        if not snapshot.next:
            raise LookupError(f"thread {thread_id} is not waiting for anything")
        values = snapshot.values
        user = User(**values["user"])
        # A dict resume value means "interrupt id -> value" to LangGraph, so wrap the hint in a string.
        hint = f"recheck:{(payload or {}).get('hint', '')}"
        return await self._drive(thread_id, Command(resume=hint), values["question"], user)

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    async def _drive(self, thread_id: str, graph_input: Any, question: str, user: User) -> RunResult:
        started = time.perf_counter()
        async with self._locks[thread_id]:
            with self.tracer.run(
                "ops-copilot", user_id=user.id, session_id=thread_id, input={"question": question, "role": user.role}
            ) as obs:
                try:
                    await self.graph.ainvoke(graph_input, self._config(thread_id))
                    result = await self.snapshot(thread_id)
                except Exception as exc:  # recorded, then surfaced to the caller as a failed run
                    result = RunResult(thread_id=thread_id, status="failed", answer=f"Run failed: {exc}")
                result.latency_ms = int((time.perf_counter() - started) * 1000)
                result.trace_url = self.tracer.trace_url()
                obs.update(output=result.model_dump(include={"status", "answer", "cost_usd", "action"}))
        self.tracer.flush()
        await self.runs.save(result, question, user)
        return result

    async def snapshot(self, thread_id: str) -> RunResult:
        snap = await self.graph.aget_state(self._config(thread_id))
        v = snap.values
        cost = run_cost(v.get("usage", []))
        waiting = bool(snap.next)
        return RunResult(
            thread_id=thread_id,
            status="awaiting_approval" if waiting else v.get("status", "completed"),
            route=v.get("route"),
            answer=_waiting_message(v.get("action")) if waiting else v.get("answer"),
            citations=v.get("citations", []),
            action=v.get("action"),
            tools_called=[t["tool"] for t in v.get("tool_log", [])],
            tool_log=v.get("tool_log", []),
            cost_usd=cost["cost_usd"],
            cost_by_model=cost["by_model"],
            input_tokens=cost["input_tokens"],
            output_tokens=cost["output_tokens"],
        )


def _waiting_message(action: dict | None) -> str:
    if not action:
        return "Waiting for input."
    return (
        f"I've proposed: {action.get('summary')}. Nothing has been written to the ERP yet; it is waiting for "
        f"approval on the ERP Bridge approval page (action {action.get('id')}). I'll continue when it is decided."
    )
