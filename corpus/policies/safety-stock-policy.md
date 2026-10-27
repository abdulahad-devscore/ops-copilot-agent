---
id: POL-INV-004
title: Inventory and Safety Stock Policy
type: policy
owner: Supply Chain Planner
access: [staff, store_manager, finance, management]
effective: 2026-04-01
version: 2
---
# Inventory and Safety Stock Policy

## ABC classification
SKUs are classified every quarter by 12-month revenue:
- **A items** (top 70% of revenue): woks, stockpots, air fryers, rice cookers, cutlery sets.
- **B items** (next 20%): frying pans, saucepans, grill pans, bakeware, rice dispensers, spice racks.
- **C items** (last 10%): airtight containers, lunch boxes, melamine plates and bowls, glass tumblers.

## Reorder rules
Every stocked SKU has a reorder rule (min/max) per warehouse in Odoo.
- **Min** = demand during replenishment lead time + safety stock.
- **Max** = min + one replenishment cycle of demand.

Default rules:
| Location | Default min | Default max |
|---|---|---|
| Retail stores (KLCC, PJ, PNG) | 10 | 40 |
| Stores, fast movers (TH-KS-4201, TH-KS-4202) | 30 | 120 |
| Stores, melamine plate TH-TW-5101 | 40 | 160 |
| Distribution centre (DC) | 80 | 400 |

Raw-material rules at the Shah Alam factory are set by the production planner, e.g. aluminium ingot min 2,000 kg / max 8,000 kg, non-stick coating min 60 L / max 250 L.

## Safety stock targets
- DC: **3 weeks** of cover for A items, **2 weeks** for B items, **1 week** for C items.
- Imported goods (Hengfa, Brightwave; 35–40 day lead time): add **2 extra weeks** of cover.
- Stores: enough for one replenishment cycle plus 3 days.

## Seasonal adjustments
Ahead of Deepavali, Hari Raya and Chinese New Year the planner raises store and DC minimums for woks and stockpots by 50% for the six weeks before the festival, and for air fryers by 30% during the year-end sale (Nov–Dec).

## Excess stock
Stock above max for more than 60 days is reviewed monthly. Options in order: transfer to a store with demand, bundle promotion, wholesale clearance at no less than cost.
