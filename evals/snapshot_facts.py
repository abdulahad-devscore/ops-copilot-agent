"""Print the ERP facts the erp_data / action eval cases depend on.

Run against the seeded stack after changing the seed, then update cases.yaml:

    python -m evals.snapshot_facts > evals/results/erp-facts.json
"""

import asyncio
import json

from ops_copilot.config import get_settings
from ops_copilot.tools import BridgeAuth, McpErpTools

QUERIES = [
    ("get_stock_by_warehouse", {"sku": "TH-SA-6104"}),
    ("get_stock_by_warehouse", {"sku": "TH-CW-2401"}),
    ("get_stock_by_warehouse", {"sku": "RM-AL-INGOT"}),
    ("get_stock_by_warehouse", {"sku": "TH-KS-4201", "warehouse": "KLCC"}),
    ("get_reorder_suggestions", {"warehouse": "FAC"}),
    ("get_reorder_suggestions", {"warehouse": "KLCC"}),
    ("get_reorder_suggestions", {"warehouse": "PNG"}),
    ("get_customer_balance", {"customer": "Dapur Bonda"}),
    ("get_customer_balance", {"customer": "Pasaraya Seri Mutiara"}),
    ("get_customer_balance", {"customer": "Restoran Selera Kampung"}),
    ("get_customer_balance", {"customer": "Borneo Living"}),
    ("list_open_invoices", {"overdue_only": True, "limit": 100}),
    ("get_sales_by_period", {"date_from": "2026-07-01", "date_to": "2026-09-30", "group_by": "channel"}),
    ("get_sales_by_period", {"date_from": "2026-07-01", "date_to": "2026-10-26", "group_by": "month"}),
    ("get_sales_by_period", {"date_from": "2026-10-01", "date_to": "2026-10-26", "group_by": "product"}),
    ("get_sales_by_period", {"date_from": "2026-09-01", "date_to": "2026-09-30", "group_by": "warehouse"}),
    ("get_sales_by_period", {"date_from": "2026-07-01", "date_to": "2026-10-26", "group_by": "customer"}),
    ("list_purchase_orders", {"status": "to_receive"}),
    ("list_purchase_orders", {"status": "to_bill"}),
    ("get_purchase_order", {"po_number": "P00009"}),
    ("get_purchase_order", {"po_number": "P00014"}),
    ("list_manufacturing_orders", {}),
    ("get_bill_of_materials", {"sku": "TH-CW-2401"}),
    ("list_transfers", {"state": "open"}),
]


async def main() -> None:
    s = get_settings()
    erp = McpErpTools(s.bridge_url, BridgeAuth(s.bridge_url, s.bridge_client_id, s.bridge_client_secret))
    out = []
    for name, args in QUERIES:
        r = await erp.call(name, args)
        out.append({"tool": name, "args": args, "is_error": r.is_error, "data": r.data if not r.is_error else r.text})
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
