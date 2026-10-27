---
id: SOP-WH-001
title: Goods Receiving and 3-Way Match
type: sop
owner: Warehouse Manager
access: [staff, store_manager, finance, management]
effective: 2026-05-01
version: 7
---
# Goods Receiving and 3-Way Match

## Receiving at the dock
1. Check the delivery order against the open purchase order in Odoo (Purchase → Receipts).
2. Count every line. Weigh bulk raw materials (aluminium ingot, resin, steel sheet) on the calibrated dock scale; tolerance ±0.5%.
3. Inspect for damage. Damaged goods go to the **QUARANTINE** location and are photographed.
4. Validate the receipt in Odoo **within 24 hours** of physical arrival. Partial deliveries are validated as received; Odoo creates a backorder for the rest.
5. Food-contact raw materials (resin, coating, silicone seals) need the supplier's certificate of analysis before release from quarantine.

## 3-way match (accounts payable)
A supplier invoice is posted only when it matches the **purchase order** (price, item) and the **goods receipt** (quantity):

- **Quantity**: invoiced quantity must not exceed quantity received. Invoicing for goods not yet received is a mismatch, even if they were ordered.
- **Price**: unit price may differ from the PO price by at most **2% or RM 50 per line, whichever is lower**.
- **Tax**: SST must match the PO line tax (raw materials for manufacturing are exempt; traded goods carry 10% SST).
- **Total**: invoice total must equal the sum of lines plus tax, within RM 1.00 rounding.
- **Duplicates**: the same supplier invoice number may not be posted twice.

Matched invoices are posted automatically by the Invoice Flow automation. Mismatches go to the AP review queue; the AP clerk resolves them with the buyer within 3 working days.

## Over-deliveries
Up to 5% over-delivery is accepted for bulk raw materials. Anything else is refused at the dock or returned at the supplier's cost.
