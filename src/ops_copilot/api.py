"""HTTP API for the copilot (consumed by the Agent Approval Console).

POST /v1/chat                          ask a question (may pause for approval)
GET  /v1/runs/{thread_id}              current state of a run
GET  /v1/runs                          recent runs with cost
POST /v1/runs/{thread_id}/resume       resume a paused run (console button)
POST /v1/bridge-callback/{thread_id}   the ERP bridge calls this when an action is decided
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ops_copilot.config import Settings, get_settings
from ops_copilot.graph import AgentDeps, build_graph
from ops_copilot.llm import AnthropicLLM
from ops_copilot.rag.embeddings import make_embedder
from ops_copilot.rag.ingest import ingest, make_store
from ops_copilot.rag.retriever import ROLE_ACCESS, Retriever, make_reranker
from ops_copilot.service import CopilotService, PgRunLog, RunResult, User
from ops_copilot.tools import BridgeAuth, McpErpTools

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    user: User
    thread_id: str | None = None


def create_app(settings: Settings | None = None, *, service: CopilotService | None = None) -> FastAPI:
    settings = settings or get_settings()
    state: dict = {"service": service}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["service"] is None:
            state["service"], cleanup = await _build_service(settings)
        else:
            cleanup = None
        yield
        if cleanup:
            await cleanup()

    app = FastAPI(
        title="Ops Copilot Agent",
        version="1.0.0",
        lifespan=lifespan,
        description="LangGraph operations agent over company documents and the ERP MCP Bridge.",
    )

    def svc() -> CopilotService:
        return state["service"]

    @app.get("/healthz", tags=["ops"])
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/chat", response_model=RunResult, tags=["chat"])
    async def chat(req: ChatRequest):
        if req.user.role not in ROLE_ACCESS:
            raise HTTPException(422, f"role must be one of {sorted(ROLE_ACCESS)}")
        return await svc().ask(req.question, req.user, req.thread_id)

    @app.get("/v1/runs", tags=["runs"])
    async def runs(limit: int = 50):
        return await svc().runs.recent(limit)

    @app.get("/v1/runs/{thread_id}", response_model=RunResult, tags=["runs"])
    async def run(thread_id: str):
        try:
            return await svc().snapshot(thread_id)
        except Exception as exc:
            raise HTTPException(404, "unknown thread") from exc

    @app.post("/v1/runs/{thread_id}/resume", response_model=RunResult, tags=["runs"])
    async def resume(thread_id: str):
        """Re-check the bridge and continue. Safe to call any time; a still-pending action just pauses again."""
        try:
            return await svc().resume(thread_id)
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/bridge-callback/{thread_id}", status_code=202, tags=["runs"])
    async def bridge_callback(
        thread_id: str, body: dict, background: BackgroundTasks, user_agent: Annotated[str | None, Header()] = None
    ):
        # The payload is only a wake-up call; the graph re-reads the decision from the bridge.
        async def go():
            try:
                await svc().resume(thread_id, {"hint": body.get("status")})
            except LookupError:
                log.info("callback for %s ignored: not waiting", thread_id)

        background.add_task(go)
        return {"accepted": True}

    return app


async def _build_service(settings: Settings):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    embedder = make_embedder(settings.embedder, settings.embedding_model)
    store = make_store(settings.vector_store, settings.database_url)
    stats = await ingest(store, embedder, settings.corpus_dir)  # idempotent; keeps the index in sync with the repo
    log.info("corpus ready: %s", stats)
    retriever = Retriever(
        store, embedder, make_reranker(settings.reranker, settings.rerank_model, settings.cohere_api_key)
    )
    erp = McpErpTools(
        settings.bridge_url, BridgeAuth(settings.bridge_url, settings.bridge_client_id, settings.bridge_client_secret)
    )
    for attempt in range(30):  # the bridge may still be starting
        try:
            await erp.list_tools()
            break
        except Exception as exc:  # noqa: BLE001
            log.info("waiting for ERP bridge (%s)", type(exc).__name__)
            await asyncio.sleep(2 + attempt)
    llm = AnthropicLLM(settings.cheap_model, settings.strong_model, settings.strong_effort)
    deps = AgentDeps(
        llm=llm,
        retriever=retriever,
        erp=erp,
        today=settings.business_date,
        callback_base=f"{settings.public_url.rstrip('/')}/v1/bridge-callback",
        max_steps=settings.max_agent_steps,
        top_k=settings.retrieve_top_k,
    )

    pool = AsyncConnectionPool(
        settings.database_url, min_size=1, max_size=10, open=False, kwargs={"autocommit": True, "prepare_threshold": 0}
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    runs = PgRunLog(pool)
    await runs.setup()
    service = CopilotService(build_graph(deps, checkpointer), runs)

    async def cleanup():
        await pool.close()
        if hasattr(store, "close"):
            await store.close()

    return service, cleanup
