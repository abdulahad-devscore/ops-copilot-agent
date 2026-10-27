"""Every branch of the agent state machine, with a scripted model and a fake ERP."""

from ops_copilot.llm import ScriptedLLM
from ops_copilot.service import User

STAFF = User(id="aisyah", role="staff")
FINANCE = User(id="daniel", role="finance")


def last_user_text(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    if isinstance(content, str):
        return content
    return " ".join(str(b.get("content", "")) for b in content)


def tool_results(messages: list[dict]) -> list[dict]:
    content = messages[-1]["content"]
    return content if isinstance(content, list) else []


async def test_knowledge_answer_with_citations(make_service):
    def script(messages, tools):
        assert "Customer Credit Policy" in messages[0]["content"]  # prefetch put the policy in context
        return "A customer goes on hold when over the limit or 60+ days overdue [S1]."

    llm = ScriptedLLM(routes={"credit hold": "knowledge"}, script=script)
    result = await make_service(llm).ask("When is a customer put on credit hold?", STAFF)
    assert result.status == "completed"
    assert result.citations[0]["doc_id"] == "POL-FIN-003"
    assert result.route == "knowledge"
    assert result.cost_by_model.keys() == {"claude-haiku-5-5", "claude-opus-5-5"}


async def test_out_of_scope_never_calls_strong_model(make_service):
    llm = ScriptedLLM(routes={"poem": "out_of_scope"})
    result = await make_service(llm).ask("Write me a poem about woks", STAFF)
    assert result.status == "completed"
    assert llm.calls == ["classify"]
    assert "can't help" in result.answer
    assert list(result.cost_by_model) == ["claude-haiku-5-5"]


async def test_data_question_uses_erp_tool(make_service, erp):
    def script(messages, tools):
        if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
            return [("get_stock_by_warehouse", {"sku": "TH-SA-6104", "warehouse": "KLCC"})]
        assert '"on_hand": 2' in last_user_text(messages)
        return "KLCC has 2 Air Fryers on hand, below the minimum of 10."

    result = await make_service(ScriptedLLM(routes={"stock": "erp_data"}, script=script)).ask(
        "How many air fryers in stock at KLCC?", STAFF
    )
    assert result.answer.startswith("KLCC has 2")
    assert result.tools_called == ["get_stock_by_warehouse"]


async def test_propose_tools_hidden_outside_action_route(make_service):
    seen = {}

    def script(messages, tools):
        seen["tools"] = {t["name"] for t in tools}
        return "ok"

    await make_service(ScriptedLLM(routes={"stock": "erp_data"}, script=script)).ask("stock of woks?", STAFF)
    assert "propose_purchase_order" not in seen["tools"]
    assert "search_documents" in seen["tools"]


async def test_action_pauses_until_bridge_decides(make_service, erp):
    def script(messages, tools):
        text = last_user_text(messages)
        if "Question:" in text:
            return [
                (
                    "propose_stock_transfer",
                    {
                        "from_warehouse": "DC",
                        "to_warehouse": "KLCC",
                        "lines": [{"sku": "TH-SA-6104", "quantity": 38}],
                        "reason": "KLCC below min",
                    },
                )
            ]
        if "APPROVED" in text:
            return "Done: transfer DC/INT/00009 was approved and created."
        if "REJECTED" in text:
            return "The transfer was rejected."
        raise AssertionError(text)

    service = make_service(ScriptedLLM(routes={"transfer": "action"}, script=script))
    first = await service.ask("Please transfer air fryers from the DC to KLCC", STAFF)
    assert first.status == "awaiting_approval"
    assert "Nothing has been written" in first.answer
    action_id = first.action["id"]
    name, args, _ = erp.calls[-1]
    assert name == "propose_stock_transfer"
    assert args["callback_url"] == f"http://ops-copilot:8100/v1/bridge-callback/{first.thread_id}"
    assert args["idempotency_key"].startswith(first.thread_id)

    # A resume while the action is still pending (e.g. a forged or early callback) changes nothing
    again = await service.resume(first.thread_id, {"status": "executed"})
    assert again.status == "awaiting_approval"

    erp.decide(action_id, "executed", odoo_ref="DC/INT/00009")
    done = await service.resume(first.thread_id)
    assert done.status == "completed"
    assert "approved" in done.answer
    assert done.action["status"] == "executed"


async def test_rejected_action_is_reported(make_service, erp):
    def script(messages, tools):
        text = last_user_text(messages)
        if "Question:" in text:
            return [
                (
                    "propose_purchase_order",
                    {
                        "supplier": "SUP-PAS",
                        "warehouse": "FAC",
                        "lines": [{"sku": "RM-AL-INGOT", "quantity": 6000}],
                        "reason": "below min",
                    },
                )
            ]
        return "Rejected by Aisyah Rahman." if "REJECTED" in text else "?"

    service = make_service(ScriptedLLM(routes={"order": "action"}, script=script))
    first = await service.ask("order more aluminium ingot", STAFF)
    erp.decide(first.action["id"], "rejected", decision_note="Use the Q1 contract")
    done = await service.resume(first.thread_id)
    assert done.answer == "Rejected by Aisyah Rahman."


async def test_bridge_validation_error_goes_back_to_model(make_service, erp):
    erp.reject_next_proposal = "No supplier matches 'Acme'"

    def script(messages, tools):
        text = last_user_text(messages)
        if "Question:" in text:
            return [
                (
                    "propose_purchase_order",
                    {
                        "supplier": "Acme",
                        "warehouse": "FAC",
                        "lines": [{"sku": "RM-AL-INGOT", "quantity": 1}],
                        "reason": "x" * 6,
                    },
                )
            ]
        assert "No supplier matches" in text
        assert tool_results(messages)[0]["is_error"] is True
        return "Acme is not an approved supplier; the approved ingot supplier is Perak Alloy Supplies."

    result = await make_service(ScriptedLLM(routes={"order": "action"}, script=script)).ask(
        "order ingot from Acme", STAFF
    )
    assert result.status == "completed"
    assert "approved supplier" in result.answer


async def test_only_one_proposal_per_step(make_service, erp):
    def script(messages, tools):
        if "Question:" in last_user_text(messages):
            line = [{"sku": "TH-SA-6104", "quantity": 5}]
            return [
                (
                    "propose_stock_transfer",
                    {"from_warehouse": "DC", "to_warehouse": "KLCC", "lines": line, "reason": "first one"},
                ),
                (
                    "propose_stock_transfer",
                    {"from_warehouse": "DC", "to_warehouse": "PJ", "lines": line, "reason": "second one"},
                ),
            ]
        return "done"

    first = await make_service(ScriptedLLM(routes={"transfer": "action"}, script=script)).ask(
        "transfer fryers to KLCC and PJ", STAFF
    )
    assert first.status == "awaiting_approval"
    assert len(erp.actions) == 1


async def test_user_token_is_forwarded(make_service, erp):
    def script(messages, tools):
        if "Question:" in last_user_text(messages):
            return [("get_customer_balance", {"customer": "Dapur Bonda"})]
        assert "Permission denied" in last_user_text(messages)
        return "You don't have access to customer balances."

    store_manager = User(id="kavitha", role="store_manager", erp_token="store-token")
    result = await make_service(ScriptedLLM(routes={"balance": "erp_data"}, script=script)).ask(
        "What is Dapur Bonda's balance?", store_manager
    )
    assert erp.calls[-1][2] == "store-token"
    assert "don't have access" in result.answer


async def test_search_documents_respects_role(make_service):
    captured = {}

    def script(messages, tools):
        if "Question:" in last_user_text(messages):
            return [("search_documents", {"query": "Dapur Bonda payment plan RM 5,000 per month"})]
        captured["text"] = last_user_text(messages)
        return "done"

    await make_service(ScriptedLLM(routes={"plan": "knowledge"}, script=script)).ask(
        "What payment plan does Dapur Bonda have?", STAFF
    )
    assert "Wholesale Customer Account Notes" not in captured["text"]

    await make_service(ScriptedLLM(routes={"plan": "knowledge"}, script=script)).ask(
        "What payment plan does Dapur Bonda have?", FINANCE
    )
    assert "Wholesale Customer Account Notes" in captured["text"]


async def test_step_limit(make_service):
    def script(messages, tools):
        return [("get_stock_by_warehouse", {"sku": "TH-SA-6104"})]

    result = await make_service(ScriptedLLM(routes={"stock": "erp_data"}, script=script), max_steps=3).ask(
        "stock loop", STAFF
    )
    assert result.status == "completed"
    assert "step limit" in result.answer
