import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ops_copilot.graph import AgentDeps, build_graph
from ops_copilot.llm import ScriptedLLM
from ops_copilot.rag.embeddings import HashingEmbedder
from ops_copilot.rag.ingest import ingest
from ops_copilot.rag.retriever import NoReranker, Retriever
from ops_copilot.rag.store import MemoryStore
from ops_copilot.service import CopilotService, MemoryRunLog
from tests.fakes import FakeErp


@pytest.fixture(scope="session")
async def retriever():
    store, embedder = MemoryStore(), HashingEmbedder()
    await ingest(store, embedder, "corpus")
    return Retriever(store, embedder, NoReranker())


@pytest.fixture
def erp():
    return FakeErp()


@pytest.fixture
def make_service(retriever, erp):
    def make(llm: ScriptedLLM, max_steps: int = 8) -> CopilotService:
        deps = AgentDeps(
            llm=llm,
            retriever=retriever,
            erp=erp,
            today="2026-10-27",
            callback_base="http://ops-copilot:8100/v1/bridge-callback",
            max_steps=max_steps,
        )
        return CopilotService(build_graph(deps, InMemorySaver()), MemoryRunLog())

    return make
