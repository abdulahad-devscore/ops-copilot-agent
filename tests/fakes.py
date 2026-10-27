"""Fake ERP bridge for agent tests."""

from __future__ import annotations

import json

from ops_copilot.tools import ToolResult

READ_TOOLS = [
    "get_stock_by_warehouse",
    "get_customer_balance",
    "list_open_invoices",
    "get_action_status",
    "get_reorder_suggestions",
]
WRITE_TOOLS = ["propose_purchase_order", "propose_stock_transfer"]


class FakeErp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.actions: dict[str, dict] = {}
        self.reject_next_proposal: str | None = None

    async def list_tools(self) -> list[dict]:
        return [
            {"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}}
            for n in READ_TOOLS + WRITE_TOOLS
        ]

    async def call(self, name: str, args: dict, token: str | None = None) -> ToolResult:
        self.calls.append((name, args, token))
        if name == "get_stock_by_warehouse":
            data = [{"sku": "TH-SA-6104", "warehouse": "KLCC", "on_hand": 2, "available": 2, "reorder_min": 10}]
        elif name == "get_customer_balance":
            if token == "store-token":
                return ToolResult("Permission denied: 'store-klcc' lacks scope 'finance:read'", True)
            data = {"customer": "Dapur Bonda Trading", "receivable": 21000, "on_credit_hold": True}
        elif name.startswith("propose_"):
            if self.reject_next_proposal:
                msg, self.reject_next_proposal = self.reject_next_proposal, None
                return ToolResult(msg, True)
            action_id = f"act-{len(self.actions) + 1}"
            data = {
                "id": action_id,
                "status": "pending",
                "summary": f"{name} {json.dumps(args.get('lines'))}",
                "action_type": name.removeprefix("propose_"),
            }
            self.actions[action_id] = data
        elif name == "get_action_status":
            data = self.actions[args["action_id"]]
        else:
            data = []
        return ToolResult(json.dumps(data), False, data)

    def decide(self, action_id: str, status: str, **extra) -> None:
        self.actions[action_id] = {**self.actions[action_id], "status": status, "decided_by": "Aisyah Rahman", **extra}
