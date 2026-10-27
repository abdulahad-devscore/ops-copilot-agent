"""HTTP API: chat, pause for approval, and auto-resume when the bridge calls back."""

from fastapi.testclient import TestClient

from ops_copilot.api import create_app
from ops_copilot.llm import ScriptedLLM
from tests.test_graph import last_user_text


def test_chat_pause_and_callback_resume(make_service, erp):
    def script(messages, tools):
        if "Question:" in last_user_text(messages):
            return [
                (
                    "propose_stock_transfer",
                    {
                        "from_warehouse": "DC",
                        "to_warehouse": "KLCC",
                        "lines": [{"sku": "TH-SA-6104", "quantity": 20}],
                        "reason": "low",
                    },
                )
            ]
        return "Transfer approved and created as DC/INT/00009."

    service = make_service(ScriptedLLM(routes={"transfer": "action"}, script=script))
    with TestClient(create_app(service=service)) as http:
        assert http.post("/v1/chat", json={"question": "hi", "user": {"id": "x", "role": "ceo"}}).status_code == 422
        run = http.post(
            "/v1/chat", json={"question": "transfer fryers to KLCC", "user": {"id": "aisyah", "role": "store_manager"}}
        ).json()
        assert run["status"] == "awaiting_approval"
        assert run["cost_usd"] > 0

        erp.decide(run["action"]["id"], "executed", odoo_ref="DC/INT/00009")
        cb = http.post(f"/v1/bridge-callback/{run['thread_id']}", json={"status": "executed"})
        assert cb.status_code == 202  # resume runs in the background, after the response

        done = http.get(f"/v1/runs/{run['thread_id']}").json()
        assert done["status"] == "completed"
        assert "DC/INT/00009" in done["answer"]
        assert http.get("/v1/runs").json()[0]["thread_id"] == run["thread_id"]
        assert http.post(f"/v1/runs/{run['thread_id']}/resume").status_code == 409  # nothing left to resume
