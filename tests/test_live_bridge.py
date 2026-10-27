"""The agent against the real ERP bridge and seeded Odoo, with a scripted model.

Proves the parts unit tests fake: OAuth + MCP over HTTP, live ERP reads, a real
proposal landing in the bridge queue, approval, the Odoo write, and the paused
run resuming. Skipped unless the bridge answers at BRIDGE_URL.

    BRIDGE_URL=http://localhost:8000 pytest tests/test_live_bridge.py
"""

import json
import os

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ops_copilot.graph import AgentDeps, build_graph
from ops_copilot.llm import ScriptedLLM
from ops_copilot.service import CopilotService, MemoryRunLog, User
from ops_copilot.tools import BridgeAuth, McpErpTools

BRIDGE = os.environ.get("BRIDGE_URL", "http://localhost:8000")


def bridge_up() -> bool:
    try:
        return httpx.get(f"{BRIDGE}/healthz", timeout=2, trust_env=False).json().get("erp") == "ok"
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not bridge_up(), reason=f"no ERP bridge at {BRIDGE}")


def token(client_id: str) -> str:
    resp = httpx.post(
        f"{BRIDGE}/oauth/token",
        trust_env=False,
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": f"{client_id}-demo-secret"},
    )
    return resp.json()["access_token"]


@pytest.fixture
def erp():
    return McpErpTools(BRIDGE, BridgeAuth(BRIDGE, "ops-copilot", "ops-copilot-demo-secret"))


def text_of(messages):
    c = messages[-1]["content"]
    return c if isinstance(c, str) else " ".join(str(b.get("content", "")) for b in c)


async def test_live_read_and_scoped_token(erp, retriever):
    tools = {t["name"] for t in await erp.list_tools()}
    assert {"get_stock_by_warehouse", "propose_purchase_order"} <= tools

    stock = await erp.call("get_stock_by_warehouse", {"sku": "TH-SA-6104", "warehouse": "KLCC"})
    assert stock.data[0]["on_hand"] == 2

    store = token("store-klcc")
    denied = await erp.call("get_customer_balance", {"customer": "Dapur Bonda"}, token=store)
    assert denied.is_error and "finance:read" in denied.text


async def test_propose_approve_resume_against_odoo(erp, retriever):
    dc = await erp.call("get_stock_by_warehouse", {"sku": "TH-SA-6104", "warehouse": "DC"})
    if dc.data[0]["available"] < 1:
        pytest.skip("DC air fryers already transferred by an earlier run; reseed to rerun")
    seen = {}

    def script(messages, tools):
        t = text_of(messages)
        if "Question:" in t:
            return [("get_stock_by_warehouse", {"sku": "TH-SA-6104"})]
        if '"warehouse": "KLCC"' in t and "propose" not in seen:
            seen["propose"] = True
            rows = json.loads(t[t.index("[") :])
            klcc = next(r for r in rows if r["warehouse"] == "KLCC")
            dc = next(r for r in rows if r["warehouse"] == "DC")
            qty = min(klcc["reorder_max"] - klcc["on_hand"], dc["available"])  # the DC can't send more than it has
            return [
                (
                    "propose_stock_transfer",
                    {
                        "from_warehouse": "DC",
                        "to_warehouse": "KLCC",
                        "lines": [{"sku": "TH-SA-6104", "quantity": qty}],
                        "reason": f"KLCC has {klcc['on_hand']:g} vs min 10; top up to max",
                    },
                )
            ]
        seen["final"] = t
        return "Done."

    deps = AgentDeps(
        llm=ScriptedLLM(routes={"transfer": "action"}, script=script), retriever=retriever, erp=erp, today="2026-10-27"
    )
    service = CopilotService(build_graph(deps, InMemorySaver()), MemoryRunLog())
    first = await service.ask("Please transfer air fryers to KLCC", User(id="it-test", role="store_manager"))
    assert first.status == "awaiting_approval", first.answer
    action = first.action
    assert action["status"] == "pending"
    assert "Transfer DC -> KLCC:" in action["summary"] and "TH-SA-6104" in action["summary"]

    approver = {"Authorization": f"Bearer {token('approver-ops')}"}
    decided = httpx.post(
        f"{BRIDGE}/api/actions/{action['id']}/approve",
        json={"note": "integration test"},
        headers=approver,
        trust_env=False,
    ).json()
    assert decided["status"] == "executed", decided
    assert decided["odoo_ref"].startswith("DC/INT/")

    done = await service.resume(first.thread_id)
    assert done.status == "completed"
    assert decided["odoo_ref"] in seen["final"] and "APPROVED" in seen["final"]
