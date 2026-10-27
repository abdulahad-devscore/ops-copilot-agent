---
id: SOP-LOG-002
title: Store Replenishment from the Distribution Centre
type: sop
owner: Logistics Supervisor
access: [staff, store_manager, finance, management]
effective: 2026-02-15
version: 6
---
# Store Replenishment from the Distribution Centre

## Schedule
Stores are replenished from the Port Klang DC on a fixed schedule:
- **KLCC and Petaling Jaya**: Tuesday and Friday. Truck leaves the DC at 06:00, arrives before store opening. Lead time **1 day**.
- **Penang**: Monday and Thursday by line-haul. Lead time **2 days**.

The cut-off for adding lines to a scheduled transfer is **14:00 the day before** departure.

## How the quantity is decided
1. Odoo reorder rules flag store SKUs whose forecast is below min.
2. The planner reviews the replenishment report and creates internal transfers DC → store up to max.
3. The DC picks, packs in returnable crates, and validates the transfer when the truck leaves.
4. The store receives and validates on arrival; discrepancies are reported in Odoo chatter within 24 hours.

## Emergency transfers
If an **A item** at a store falls below **25% of its min** (for example, fewer than 3 air fryers against a min of 10), the store manager may request a same-day emergency transfer from the DC. Emergency transfers to KLCC and PJ go by courier (cost ~RM 45 per crate) and need the Logistics Supervisor's approval. Penang emergencies ship next-day air freight only for A items.

Store-to-store transfers (e.g. PJ → KLCC) are allowed when the DC is out of stock; the sending store must stay above its own min.

## Ops Copilot
The copilot may propose transfers. Every proposal goes to the approval page; the Logistics Supervisor or Operations Manager approves.
