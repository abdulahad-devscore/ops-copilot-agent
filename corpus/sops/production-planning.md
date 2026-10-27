---
id: SOP-MFG-001
title: Production Planning and Manufacturing Orders
type: sop
owner: Production Manager
access: [staff, finance, management]
effective: 2026-03-15
version: 4
---
# Production Planning and Manufacturing Orders

## Weekly plan
The production plan for the Shah Alam factory is fixed every **Thursday** for the following week. Inputs: DC reorder suggestions for made-in-house SKUs, confirmed wholesale orders, festival forecasts.

## Lines and capacity
| Line | Products | Capacity per shift |
|---|---|---|
| Casting line | Woks, stockpots, saucepans, grill pans, frying pans | 450 units |
| Pressing line | Bakeware (loaf tins, cake tins, muffin trays, baking trays) | 600 units |
| Injection moulding | Airtight containers, lunch boxes, rice dispensers, spice racks | 2,500 units |

The factory runs two shifts, Monday to Saturday. Changeover between wok sizes takes 45 minutes; changeover on the injection line (colour or mould) takes 2 hours, so storage items are batched in runs of at least 1,000 units.

## Releasing a manufacturing order
1. Confirm the MO in Odoo only when **components are available** (Odoo shows "Available"). Partial availability needs the Production Manager's approval.
2. Non-stick coated items need the coating line's cure oven at **400 °C** for 12 minutes; book oven time with the MO.
3. Quality control samples finished goods at **AQL 2.5** (General Inspection Level II). Failed lots are quarantined and reworked or scrapped.
4. Mark the MO done when finished goods are moved to the finished-goods area; transfer to the DC daily at 16:00.

## Minimum batch sizes
- Cookware: 100 units per SKU.
- Bakeware: 200 units.
- Storage items: 1,000 units.

## Ops Copilot
The copilot may propose manufacturing orders. The Production Manager approves on the ERP Bridge approval page after checking line capacity.
