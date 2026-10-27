"""Generate the long tail of the Tembusu document corpus.

The hand-written policies and SOPs in corpus/{policies,sops,pricing,finance,management}
carry the facts most questions are about. This script adds ~180 realistic
supporting documents (product sheets, QC specs, supplier and site profiles,
meeting minutes, incident reports, staff FAQs, system how-tos) so retrieval
has to find the right needle in a realistic haystack.

Deterministic: same output every run. Usage: python scripts/generate_corpus.py
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "corpus" / "generated"
RNG = random.Random(7)
ALL = "[staff, store_manager, finance, management]"

# sku, name, category, list price, source, supplier, warranty
PRODUCTS = [
    ("TH-CW-2401", "Cast Aluminium Wok 32cm", "Cookware", 129.00, "make", None, "5 years body, 2 years coating"),
    ("TH-CW-2402", "Cast Aluminium Wok 36cm", "Cookware", 149.00, "make", None, "5 years body, 2 years coating"),
    ("TH-CW-2410", "Non-stick Frying Pan 24cm", "Cookware", 79.00, "make", None, "2 years coating"),
    ("TH-CW-2411", "Non-stick Frying Pan 28cm", "Cookware", 95.00, "make", None, "2 years coating"),
    ("TH-CW-2420", "Stockpot 24cm with Glass Lid", "Cookware", 159.00, "make", None, "5 years body"),
    ("TH-CW-2421", "Stockpot 28cm with Glass Lid", "Cookware", 189.00, "make", None, "5 years body"),
    ("TH-CW-2430", "Saucepan 18cm", "Cookware", 69.00, "make", None, "2 years coating"),
    ("TH-CW-2440", "Grill Pan 28cm", "Cookware", 109.00, "make", None, "2 years coating"),
    ("TH-BW-3101", "Loaf Tin 900g", "Bakeware", 29.90, "make", None, "1 year"),
    ("TH-BW-3102", "Round Cake Tin 20cm", "Bakeware", 34.90, "make", None, "1 year"),
    ("TH-BW-3103", "Muffin Tray 12-cup", "Bakeware", 39.90, "make", None, "1 year"),
    ("TH-BW-3104", "Baking Tray 40x30cm", "Bakeware", 32.90, "make", None, "1 year"),
    ("TH-KS-4201", "Airtight Container 0.8L", "Kitchen Storage", 12.90, "make", None, "1 year"),
    ("TH-KS-4202", "Airtight Container 1.5L", "Kitchen Storage", 16.90, "make", None, "1 year"),
    ("TH-KS-4203", "Airtight Container 3.0L", "Kitchen Storage", 22.90, "make", None, "1 year"),
    ("TH-KS-4210", "Rice Dispenser 10kg", "Kitchen Storage", 69.00, "make", None, "1 year"),
    ("TH-KS-4220", "Lunch Box 3-compartment", "Kitchen Storage", 19.90, "make", None, "1 year"),
    ("TH-KS-4230", "Spice Rack 12-jar", "Kitchen Storage", 49.90, "make", None, "1 year"),
    ("TH-TW-5101", "Melamine Dinner Plate 10in", "Tableware", 8.90, "buy", "Hengfa Tableware Trading", "none"),
    ("TH-TW-5102", "Melamine Bowl 6in", "Tableware", 6.90, "buy", "Hengfa Tableware Trading", "none"),
    ("TH-TW-5110", "Stainless Cutlery Set 24pc", "Tableware", 59.90, "buy", "Hengfa Tableware Trading", "2 years"),
    ("TH-TW-5120", "Glass Tumbler Set of 6", "Tableware", 29.90, "buy", "Johor Glassworks", "none"),
    ("TH-SA-6101", "Electric Kettle 1.7L", "Small Appliances", 89.00, "buy", "Brightwave Home Appliances", "12 months"),
    ("TH-SA-6102", "Rice Cooker 1.8L", "Small Appliances", 159.00, "buy", "Brightwave Home Appliances", "12 months"),
    ("TH-SA-6103", "Hand Blender 300W", "Small Appliances", 119.00, "buy", "Brightwave Home Appliances", "12 months"),
    ("TH-SA-6104", "Air Fryer 4.5L", "Small Appliances", 299.00, "buy", "Brightwave Home Appliances", "12 months"),
]

CARTON = {"Cookware": 6, "Bakeware": 12, "Kitchen Storage": 24, "Tableware": 48, "Small Appliances": 4}
MATERIAL = {
    "TH-CW": "Cast aluminium A356 body, bakelite handles",
    "TH-BW": "0.8 mm carbon steel with 2-layer non-stick coating",
    "TH-KS": "Food-grade polypropylene (BPA-free), silicone seal",
    "TH-TW-51": "Melamine (not microwave safe)",
    "TH-TW-5110": "18/0 stainless steel",
    "TH-TW-5120": "Tempered soda-lime glass",
    "TH-SA": "See manufacturer's manual; 240V, UK 3-pin plug, SIRIM certified",
}
CARE = {
    "Cookware": "Hand wash with a soft sponge. Use wooden or silicone utensils. Never heat an empty non-stick pan on high "
    "heat. Season cast aluminium woks with a thin film of oil after washing.",
    "Bakeware": "Hand wash recommended. Cool before washing. Do not use metal scourers on the coated surface.",
    "Kitchen Storage": "Dishwasher safe (top rack). Microwave safe without the lid. Freezer safe to -20 °C.",
    "Tableware": "Dishwasher safe. Melamine items must not be used in the microwave or oven.",
    "Small Appliances": "Unplug before cleaning. Wipe the body with a damp cloth; do not immerse. Removable parts "
    "(basket, jug lid, blender shaft) are hand wash only.",
}

RAW = [
    ("RM-AL-INGOT", "Aluminium Ingot A356", "kg", "Perak Alloy Supplies", "Si 6.5–7.5%, Mg 0.25–0.45%, Fe max 0.2%"),
    ("RM-NS-COAT", "Non-stick Coating (PTFE, 2-layer)", "L", "Selangor Coatings", "PFOA-free, cure 400 °C for 12 min"),
    ("RM-HDL-BAK", "Bakelite Handle", "unit", "Klang Valley Plastics", "Heat resistant to 180 °C"),
    ("RM-LID-GL24", "Tempered Glass Lid 24cm", "unit", "Johor Glassworks", "Steam vent, stainless rim"),
    ("RM-LID-GL28", "Tempered Glass Lid 28cm", "unit", "Johor Glassworks", "Steam vent, stainless rim"),
    ("RM-STL-SHEET", "Carbon Steel Sheet 0.8mm", "kg", "Butterworth Steel Trading", "SPCC grade, oiled"),
    ("RM-PP-RESIN", "Polypropylene Resin (food grade)", "kg", "Klang Valley Plastics", "MFI 12, FDA 21 CFR 177.1520"),
    ("RM-SIL-SEAL", "Silicone Seal Ring", "unit", "Klang Valley Plastics", "Food-grade platinum-cured silicone"),
    ("RM-PKG-BOX", "Printed Carton", "unit", "Penang Packaging", "5-ply, 2-colour flexo print"),
    ("RM-PKG-SLV", "Printed Sleeve", "unit", "Penang Packaging", "300 gsm art card, matt laminate"),
]

SUPPLIERS = [
    ("Perak Alloy Supplies Sdn Bhd", "SUP-PAS", "Ipoh, Perak", "aluminium ingot", "ISO 9001", "Encik Faizal Hamid"),
    (
        "Selangor Coatings Sdn Bhd",
        "SUP-SCS",
        "Klang, Selangor",
        "non-stick coatings",
        "ISO 9001, FDA food contact",
        "Ms. Lim Pei Shan",
    ),
    (
        "Klang Valley Plastics Sdn Bhd",
        "SUP-KVP",
        "Rawang, Selangor",
        "resin, seals and handles",
        "ISO 9001",
        "Mr. Raj Kumar",
    ),
    (
        "Johor Glassworks Sdn Bhd",
        "SUP-JGW",
        "Pasir Gudang, Johor",
        "tempered glass",
        "ISO 9001, SIRIM",
        "Ms. Nurul Izzah",
    ),
    (
        "Penang Packaging Sdn Bhd",
        "SUP-PPK",
        "Bayan Lepas, Penang",
        "cartons and sleeves",
        "FSC chain of custody",
        "Mr. Tan Wei Ming",
    ),
    (
        "Butterworth Steel Trading Sdn Bhd",
        "SUP-BST",
        "Butterworth, Penang",
        "steel sheet",
        "ISO 9001",
        "Mr. Goh Chee Keong",
    ),
    (
        "Hengfa Tableware Trading Co., Ltd.",
        "SUP-HTT",
        "Guangzhou, China",
        "melamine and cutlery",
        "BSCI audit, LFGB",
        "Ms. Chen Li",
    ),
    (
        "Brightwave Home Appliances Co., Ltd.",
        "SUP-BHA",
        "Shenzhen, China",
        "small appliances",
        "CB scheme, SIRIM via local agent",
        "Mr. Kevin Zhou",
    ),
]

WAREHOUSES = [
    (
        "FAC",
        "Shah Alam Factory",
        "Lot 12, Jalan Perindustrian 7, 40000 Shah Alam",
        "Mon–Sat 07:00–23:00 (two shifts)",
        6,
        "Raw materials, work in progress, finished goods awaiting transfer",
        "Encik Hafiz (Production Manager)",
    ),
    (
        "DC",
        "Port Klang Distribution Centre",
        "No. 8, Jalan Kemajuan, Pulau Indah, 42920 Port Klang",
        "Mon–Sat 06:00–20:00",
        10,
        "4,200 pallet positions; online fulfilment mezzanine; returns cage",
        "Ms. Priya Nair (Warehouse Manager)",
    ),
    (
        "KLCC",
        "Store - KLCC",
        "Lot 3-21, Level 3, Suria KLCC, Kuala Lumpur",
        "Daily 10:00–22:00",
        0,
        "Back room of 40 m², 180 shelf bays",
        "Ms. Aisyah Zulkifli (Store Manager)",
    ),
    (
        "PJ",
        "Store - Petaling Jaya",
        "G-45, Ground Floor, 1 Utama Shopping Centre, Petaling Jaya",
        "Daily 10:00–22:00",
        0,
        "Back room of 30 m²",
        "Mr. Jason Lee (Store Manager)",
    ),
    (
        "PNG",
        "Store - Penang",
        "2-08, Gurney Plaza, Georgetown, Penang",
        "Daily 10:00–22:00",
        0,
        "Back room of 25 m²",
        "Ms. Kavitha Rao (Store Manager)",
    ),
]

CUSTOMERS = [
    (
        "Pasaraya Seri Mutiara Sdn Bhd",
        "CUS-PSM",
        "Supermarket chain, 22 branches in the Klang Valley and Perak",
        "Delivers to central warehouse in Shah Alam; booking via their vendor portal 48 hours ahead",
        "Mondays",
    ),
    (
        "HomeMart Ipoh Sdn Bhd",
        "CUS-HMI",
        "Home-improvement stores in Perak (6 outlets)",
        "Pallets to Ipoh DC, Lintas Utara Haulage; tail-lift required",
        "every 10 days",
    ),
    (
        "Dapur Bonda Trading",
        "CUS-DBT",
        "Kitchenware shops in Selangor (3 outlets)",
        "Deliver to Kajang shop; no deliveries on Fridays 12:00–14:30",
        "fortnightly",
    ),
    (
        "Kuching Home Centre Sdn Bhd",
        "CUS-KHC",
        "Distributor for Sarawak",
        "Ex-Port Klang, Borneo Sea Link LCL; consolidate with Borneo Living when possible",
        "monthly",
    ),
    (
        "Melaka Hardware & Kitchen Enterprise",
        "CUS-MHK",
        "Hardware and kitchen shop, Melaka",
        "Selatan Express pallet; shop closes at 18:00",
        "every 2 weeks",
    ),
    (
        "Hotel Cempaka Kuala Lumpur",
        "CUS-HCK",
        "5-star hotel, 420 rooms, 3 restaurants",
        "Loading bay B, Jalan Ampang; security pass needed; PO number on every DO",
        "twice a quarter",
    ),
    (
        "Restoran Selera Kampung Group",
        "CUS-RSK",
        "Restaurant group, 5 outlets in KL",
        "Deliver to central kitchen in Sentul before 10:00",
        "monthly",
    ),
    (
        "Borneo Living Retail Sdn Bhd",
        "CUS-BLR",
        "Home-store chain in Sabah (9 stores)",
        "Ex-Port Klang, sea freight to Kota Kinabalu; container load planned with their buyer",
        "every 3 weeks",
    ),
]


def front(doc_id: str, title: str, doc_type: str, owner: str, effective: date, access: str = ALL) -> str:
    return (
        f"---\nid: {doc_id}\ntitle: {json.dumps(title)}\ntype: {doc_type}\nowner: {owner}\naccess: {access}\n"
        f"effective: {effective.isoformat()}\nversion: 1\n---\n"
    )


def write(path: str, text: str) -> None:
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n")


def material_for(sku: str) -> str:
    for prefix in ("TH-TW-5110", "TH-TW-5120", "TH-TW-51", "TH-CW", "TH-BW", "TH-KS", "TH-SA"):
        if sku.startswith(prefix):
            return MATERIAL[prefix]
    return ""


def product_docs() -> None:
    for sku, name, cat, price, source, supplier, warranty in PRODUCTS:
        origin = "Made in Malaysia at the Shah Alam factory" if source == "make" else f"Imported from {supplier}"
        weight = round(RNG.uniform(0.2, 4.5), 2)
        write(
            f"products/{sku}-datasheet.md",
            front(f"PRD-{sku}", f"Product Sheet: {name} ({sku})", "product_sheet", "Product Manager", date(2026, 1, 15))
            + f"""
# {name} ({sku})

| Field | Value |
|---|---|
| Category | {cat} |
| Retail list price | RM {price:.2f} (excl. SST) |
| Origin | {origin} |
| Material | {material_for(sku)} |
| Net weight | {weight} kg |
| Master carton | {CARTON[cat]} units |
| Warranty | {warranty} |

## Selling points
- {name} designed for Malaysian home kitchens{" and high-heat stir frying" if "Wok" in name else ""}.
- {"Even heat distribution from a thick cast base." if cat == "Cookware" else "Easy to clean and store."}
- Backed by Tembusu after-sales service at all stores.

## Care
{CARE[cat]}
""",
        )
        faq_q = {
            "Cookware": (
                "Can I use it on an induction hob?",
                "No. The cast aluminium base does not work on induction. Gas, electric coil and ceramic hobs are fine.",
            ),
            "Bakeware": ("What is the maximum oven temperature?", "230 °C. Above that the coating may discolour."),
            "Kitchen Storage": (
                "Is it airtight for rice and flour?",
                "Yes, the silicone seal keeps out moisture and weevils when the four clips are closed.",
            ),
            "Tableware": ("Can it go in the microwave?", "Glass and stainless items yes for reheating; melamine no."),
            "Small Appliances": (
                "What plug does it have?",
                "UK 3-pin, 240V, SIRIM certified. Warranty is 12 months through Brightwave; keep the receipt.",
            ),
        }[cat]
        write(
            f"products/{sku}-faq.md",
            front(f"FAQ-{sku}", f"Customer FAQ: {name}", "faq", "Customer Service", date(2026, 3, 1))
            + f"""
# Customer FAQ: {name} ({sku})

**{faq_q[0]}**
{faq_q[1]}

**Is there a warranty?**
{warranty if warranty != "none" else "No product warranty; defects on delivery are replaced within the returns period."}

**Where is it stocked?**
All three stores (KLCC, Petaling Jaya, Penang) and online. Check live stock with the Ops Copilot or Odoo before
promising a customer a reservation.
""",
        )
        if source == "make":
            aql = "2.5" if cat == "Cookware" else "4.0"
            write(
                f"quality/{sku}-qc-spec.md",
                front(f"QC-{sku}", f"QC Inspection Spec: {name}", "qc_spec", "QA Manager", date(2026, 2, 1))
                + f"""
# QC Inspection Spec: {name} ({sku})

Sampling: AQL {aql}, General Inspection Level II, per production lot.

| Check | Method | Accept |
|---|---|---|
| Dimensions | Calipers / tape vs drawing {sku}-D1 | ±1.0 mm |
| Visual | Under 1,000 lux at 40 cm | No pits, bubbles or coating peel |
| {"Coating adhesion" if cat != "Kitchen Storage" else "Seal leak test"} | {"Cross-hatch tape test" if cat != "Kitchen Storage" else "Water inversion 30 s"} | {"Class 0–1" if cat != "Kitchen Storage" else "No leak"} |
| Packaging | Barcode scan, carton count | {CARTON[cat]} per carton, EAN readable |

Critical defects (sharp edges, loose handles, cracked lids) reject the lot. Rework is allowed once; the second failure
means scrap and a corrective action report to the Production Manager.
""",
            )


def raw_docs() -> None:
    for code, name, uom, supplier, spec in RAW:
        write(
            f"materials/{code}.md",
            front(f"MAT-{code}", f"Material Spec: {name}", "material_spec", "QA Manager", date(2026, 1, 20))
            + f"""
# Material Spec: {name} ({code})

- Unit of measure: {uom}
- Approved supplier: {supplier}
- Specification: {spec}
- Storage: Shah Alam factory raw-material store, FIFO by lot; keep dry and off the floor.
- Incoming inspection: certificate of analysis checked against spec on every lot; {"weigh on dock scale (±0.5%)" if uom == "kg" else "count by carton"}.
- Used in: {"cookware casting" if "Alumin" in name else "see bills of materials in Odoo"}.
""",
        )


def supplier_docs() -> None:
    for name, ref, city, scope, certs, contact in SUPPLIERS:
        otif = RNG.choice([91.5, 93.0, 95.5, 97.2, 98.1, 88.4])
        write(
            f"suppliers/{ref}.md",
            front(
                f"SUP-PROFILE-{ref}",
                f"Supplier Profile: {name}",
                "supplier_profile",
                "Procurement Lead",
                date(2026, 7, 1),
            )
            + f"""
# Supplier Profile: {name} ({ref})

- Location: {city}
- Supplies: {scope}
- Certifications: {certs}
- Account contact: {contact}
- On-time-in-full, last 12 months: {otif}%
- Invoices: emailed as PDF to ap@tembusu-housewares.example; must quote our PO number (e.g. P00012).

## Notes
{"Below the 95% OTIF target; corrective action plan requested in September 2026." if otif < 95 else "Meets the scorecard targets."}
Delivery address for raw materials and packaging is the Shah Alam factory; traded goods go to the Port Klang DC.
""",
        )


def site_docs() -> None:
    for code, name, address, hours, docks, capacity, contact in WAREHOUSES:
        write(
            f"sites/{code}.md",
            front(
                f"SITE-{code}", f"Site Profile: {name} ({code})", "site_profile", "Operations Manager", date(2026, 1, 5)
            )
            + f"""
# Site Profile: {name} (warehouse code {code})

- Address: {address}
- Operating hours: {hours}
- Loading docks: {docks if docks else "none (deliveries via mall loading bay)"}
- Capacity: {capacity}
- Site lead: {contact}
""",
        )
    store_notes = {
        "KLCC": (
            "tourists and office workers; peak 12:00–14:00 and after 18:00",
            "Mall loading bay closes at 09:30; trucks must arrive before then.",
        ),
        "PJ": (
            "families at weekends; mall renovation on level G until December 2026",
            "Use loading bay 3; bay 1 is closed for renovation.",
        ),
        "PNG": (
            "weekend shoppers and tourists; weekdays are quiet",
            "Line-haul arrives Monday and Thursday after 14:00; unload within 1 hour.",
        ),
    }
    for code, (customers, delivery) in store_notes.items():
        write(
            f"stores/{code}-playbook.md",
            front(
                f"STORE-{code}",
                f"Store Playbook: {code}",
                "store_playbook",
                "Head of Retail",
                date(2026, 6, 1),
                "[staff, store_manager, management]",
            )
            + f"""
# Store Playbook: {code}

- Customer profile: {customers}.
- Deliveries: {delivery}
- Opening checklist: count cash float, check campaign shelf talkers, review the replenishment report for SKUs below min.
- Closing checklist: end-of-day Z report, move damaged items to QUARANTINE, lock the appliance cage.
- Staffing: 1 manager, 1 senior sales associate, 3–4 part-timers on weekends.
- Display: woks and stockpots at the entrance table during festivals; air fryers demo-cooked on Saturdays.
""",
        )


def customer_docs() -> None:
    for name, ref, profile, delivery, cadence in CUSTOMERS:
        write(
            f"customers/{ref}.md",
            front(
                f"CUS-PROFILE-{ref}", f"Customer Profile: {name}", "customer_profile", "Head of Sales", date(2026, 5, 1)
            )
            + f"""
# Customer Profile: {name} ({ref})

- Business: {profile}
- Ordering cadence: {cadence}
- Delivery instructions: {delivery}
- Price terms: wholesale (15% off list; +3% above RM 20,000 per order)
- For credit and payment questions see Finance; account notes are confidential.
""",
        )


def minutes() -> None:
    start = date(2026, 8, 3)
    topics = [
        "Air fryer demand above forecast; KLCC stock-outs on weekends.",
        "Resin short delivery from Klang Valley Plastics; injection line rescheduled.",
        "Deepavali campaign plan: wok and stockpot minimums raised 50%.",
        "Dapur Bonda overdue balance; payment plan proposed by Finance.",
        "Penang store below target; footfall down 8%.",
        "Hengfa partial shipment of melamine; next vessel in 2 weeks.",
        "Cycle count variance on hand blenders at PJ (3 units).",
        "New ParcelGo pickup time 16:30 from next Monday.",
        "Aluminium ingot price up 6%; procurement to negotiate Q1 2027 contract.",
        "Forklift near miss at DC aisle 14; refresher training scheduled.",
        "Online orders up 22% month on month; extra packer on the 15:00 wave.",
        "Invoice Flow automation go-live for supplier invoices; AP review queue owner confirmed.",
    ]
    for i, topic in enumerate(topics):
        d = start + timedelta(days=7 * i)
        write(
            f"minutes/ops-weekly-{d.isoformat()}.md",
            front(
                f"MIN-{d.isoformat()}",
                f"Weekly Ops Meeting {d:%d %B %Y}",
                "minutes",
                "Operations Manager",
                d,
                "[staff, store_manager, finance, management]",
            )
            + f"""
# Weekly Operations Meeting, {d:%A %d %B %Y}

Attendees: Operations Manager, Warehouse Manager, Production Manager, Head of Retail, Finance Controller, Procurement Lead.

## Main topic
{topic}

## Actions
- Owner to update the plan in Odoo by {(d + timedelta(days=3)):%d %b}.
- Review again at next week's meeting.

## KPIs this week
- DC on-time dispatch: {RNG.uniform(94, 99.5):.1f}%
- Online orders shipped same day: {RNG.uniform(90, 99):.1f}%
- Factory output vs plan: {RNG.uniform(88, 104):.0f}%
""",
        )


def incidents() -> None:
    items = [
        (
            "Forklift near miss in DC aisle 14",
            "DC",
            "No injury. Pedestrian walked outside the green walkway. Refresher briefing held.",
        ),
        (
            "Pallet of glass tumblers dropped",
            "DC",
            "48 sets damaged (RM 504 at cost). Scrapped with Warehouse Manager approval; carrier not involved.",
        ),
        (
            "Air fryer stock-out at KLCC",
            "KLCC",
            "Saturday demo sold 9 units; store had 2 left by Sunday. Emergency transfer requested.",
        ),
        (
            "Resin short delivery",
            "FAC",
            "Klang Valley Plastics delivered 800 of 1,000 kg. Backorder of 200 kg open; injection line moved to bakeware.",
        ),
        (
            "Cure oven temperature alarm",
            "FAC",
            "Oven dropped to 370 °C for 6 minutes; 120 frying pans quarantined and re-tested; 8 scrapped.",
        ),
        (
            "Wrong price label on cutlery sets",
            "PJ",
            "Campaign price RM 49.90 shown before campaign start; honoured for 3 customers.",
        ),
        (
            "Late line-haul to Penang",
            "PNG",
            "Truck breakdown; delivery arrived Tuesday instead of Monday. No stock-outs.",
        ),
        (
            "Customer injury claim (burn)",
            "KLCC",
            "Customer touched demo air fryer basket. First aid given; demo area now has a heat warning sign.",
        ),
        (
            "ParcelGo lost parcel",
            "DC",
            "Online order of a rice cooker lost in transit; claim filed within 3 days; customer re-shipped.",
        ),
        (
            "Shrinkage on hand blenders",
            "PJ",
            "Cycle count found 3 units missing; CCTV review requested; reported to Head of Retail.",
        ),
        (
            "Mislabelled cartons",
            "FAC",
            "500 sleeves for TH-KS-4202 printed with the 1.5L barcode on a 0.8L sleeve; reprinted by Penang Packaging at their cost.",
        ),
        ("Power outage at DC", "DC", "45-minute outage; online pick wave delayed; all orders still shipped same day."),
        (
            "Water leak in KLCC back room",
            "KLCC",
            "Ceiling leak damaged 6 carton sleeves; product undamaged; mall management notified.",
        ),
        (
            "Supplier invoice duplicated",
            "DC",
            "Johor Glassworks sent the same invoice twice; second copy rejected by the duplicate check.",
        ),
        (
            "Wholesale delivery refused",
            "DC",
            "Hotel Cempaka refused a delivery without their PO number on the DO; redelivered next day.",
        ),
    ]
    d = date(2026, 7, 6)
    for i, (title, site, body) in enumerate(items):
        d += timedelta(days=RNG.randint(3, 8))
        write(
            f"incidents/INC-2026-{i + 101}.md",
            front(f"INC-2026-{i + 101}", f"Incident Report: {title}", "incident", "HSE Officer", d)
            + f"""
# Incident Report INC-2026-{i + 101}: {title}

- Date: {d:%d %B %Y}
- Site: {site}
- Description: {body}
- Reported by: site lead
- Status: closed
""",
        )


def staff_faqs() -> None:
    faqs = [
        (
            "Can a customer return an air fryer they opened but did not like?",
            "No. Small appliances cannot be returned for change of mind once the seal is broken. Defects within 7 days "
            "are exchanged in store; after that the 12-month Brightwave warranty applies.",
        ),
        ("How long do members have to return items?", "30 days with a receipt; non-members have 14 days."),
        ("What discount can a store manager give?", "Up to 10% per transaction, capped at RM 300."),
        ("When does the replenishment truck come to KLCC?", "Tuesday and Friday mornings before opening."),
        (
            "How do I request an emergency transfer?",
            "Ask the Ops Copilot or the Logistics Supervisor when an A item drops below 25% of its min. Emergency "
            "transfers need the Logistics Supervisor's approval.",
        ),
        ("Is shipping free for online orders?", "Free in West Malaysia from RM 150; East Malaysia is RM 25 flat."),
        (
            "Can I combine a voucher with the Deepavali air fryer price?",
            "No, vouchers cannot be combined with campaign prices on small appliances.",
        ),
        (
            "What is the staff discount?",
            "20% off list up to RM 1,000 a year; small appliances excluded during campaigns.",
        ),
        ("Who approves a refund above RM 1,000?", "The store manager."),
        ("Where do damaged returns go?", "The store's QUARANTINE location, then to the DC returns cage on Friday."),
        ("Do woks work on induction?", "No, the cast aluminium base is not induction compatible."),
        ("What warranty does the non-stick coating have?", "Two years under normal domestic use."),
        (
            "How do I check stock at another store?",
            "Use the Ops Copilot (if your role allows) or Odoo Inventory → Reporting → Stock, filtered by warehouse.",
        ),
        ("What time do stores open?", "All three stores open daily 10:00–22:00."),
        ("How are cash refunds above RM 500 handled?", "By bank transfer within 5 working days."),
        ("Who can approve a purchase order of RM 30,000?", "The Operations Manager and the Finance Controller."),
        ("What is the cut-off to add items to a scheduled transfer?", "14:00 the day before the truck leaves."),
        ("When are fire drills held?", "Twice a year at each site."),
        ("How quickly must a delivery discrepancy be reported?", "Within 24 hours in Odoo chatter."),
        (
            "Can I move stock from PJ to KLCC?",
            "Yes, store-to-store transfers are allowed when the DC is out of stock, "
            "as long as the sending store stays above its own min.",
        ),
    ]
    for i, (q, a) in enumerate(faqs, 1):
        write(
            f"staff-faq/faq-{i:02d}.md",
            front(f"STAFF-FAQ-{i:02d}", f"Staff FAQ: {q}", "faq", "Head of Retail", date(2026, 6, 15))
            + f"# {q}\n\n{a}\n",
        )


def howtos() -> None:
    items = [
        (
            "Approve an agent proposal",
            "Open the ERP Bridge approval page (http://localhost:8000/approvals), sign in, "
            "review the lines and reason, then Approve or Reject. Approval writes to Odoo immediately and is logged.",
        ),
        (
            "Create an internal transfer in Odoo",
            "Inventory → Operations → Transfers → New. Choose the DC internal "
            "operation type, set source DC/Stock and destination e.g. KLCC/Stock, add lines, Mark as To Do.",
        ),
        (
            "Validate a receipt",
            "Inventory → Receipts → open the receipt → enter quantities → Validate. Partial "
            "receipts create a backorder automatically.",
        ),
        ("Run the aged receivables report", "Accounting → Reporting → Aged Receivable; filter by partner."),
        (
            "Check a customer's credit status",
            "Open the customer in Odoo (Contacts) → Invoicing tab shows credit limit "
            "and total receivable. The Ops Copilot can also report credit-hold status.",
        ),
        ("Raise a credit note", "Open the posted invoice → Credit Note → reason and amount → Post. Needs Finance."),
        ("Print shelf labels", "Inventory → Products → select → Print Labels; choose 'Shelf label 50x30'."),
        ("Reset your Odoo password", "Use 'Reset password' on the login page; IT approves within one working day."),
        (
            "Ask the Ops Copilot a question",
            "Open the Approval Console chat. The copilot answers from policies and live "
            "ERP data and cites its sources. It can draft purchase orders, transfers and manufacturing orders, but a "
            "person must approve them.",
        ),
        ("Report a near miss", "Fill the HSE log form on the intranet within the same shift."),
    ]
    for i, (title, body) in enumerate(items, 1):
        write(
            f"howto/howto-{i:02d}.md",
            front(f"HOWTO-{i:02d}", f"How to: {title}", "howto", "IT", date(2026, 4, 1))
            + f"# How to: {title}\n\n{body}\n",
        )


def training() -> None:
    modules = [
        ("Receiving and quality basics", "Counting, weighing, quarantine, certificates of analysis."),
        ("Store opening and closing", "Cash float, campaign displays, replenishment report, Z report."),
        ("Customer service and returns", "Returns windows, appliance rules, warranty claims."),
        ("Forklift safety refresher", "Walkways, stacking heights, licence renewal every 2 years."),
        ("Odoo for store managers", "Stock reports, transfers, discrepancy reporting."),
        ("Working with the Ops Copilot", "What it can answer, how it cites sources, why every write needs approval."),
    ]
    for i, (title, body) in enumerate(modules, 1):
        write(
            f"training/module-{i}.md",
            front(f"TRN-{i:02d}", f"Training Module {i}: {title}", "training", "HR", date(2026, 2, 1))
            + f"# Training Module {i}: {title}\n\nCovers: {body}\n"
            "Duration: 45 minutes. Completion is recorded in the HR system.\n",
        )


def category_docs() -> None:
    guides = {
        "Cookware": "Woks and stockpots are A items and festival best sellers; frying pans and saucepans are B items.",
        "Bakeware": "Sales peak before Hari Raya and Christmas baking seasons; all items are B class.",
        "Kitchen Storage": "High-volume, low-price C items (except rice dispensers and spice racks, which are B).",
        "Tableware": "Imported from Hengfa (melamine, cutlery) and Johor Glassworks (tumblers); HoReCa buys in bulk.",
        "Small Appliances": "Imported from Brightwave; lowest margin category; air fryers are the fastest-growing SKU.",
    }
    for cat, note in guides.items():
        slug = cat.lower().replace(" ", "-")
        skus = ", ".join(f"{p[0]} {p[1]}" for p in PRODUCTS if p[2] == cat)
        write(
            f"categories/{slug}-guide.md",
            front(
                f"CAT-{slug.upper()}", f"Category Guide: {cat}", "category_guide", "Category Manager", date(2026, 3, 10)
            )
            + f"""
# Category Guide: {cat}

{note}

SKUs: {skus}.

Master carton: {CARTON[cat]} units. Care: {CARE[cat]}
""",
        )
        write(
            f"packaging/{slug}-packaging.md",
            front(
                f"PKG-{slug.upper()}", f"Packaging Standard: {cat}", "packaging_spec", "QA Manager", date(2026, 2, 20)
            )
            + f"""
# Packaging Standard: {cat}

- Unit packaging: {"printed carton (RM-PKG-BOX)" if cat in ("Cookware",) else "printed sleeve (RM-PKG-SLV)" if cat in ("Bakeware", "Kitchen Storage") else "supplier retail box"}.
- Master carton: {CARTON[cat]} units, 5-ply, maximum 18 kg gross.
- Pallet: 1.2 x 1.0 m, maximum height 1.5 m, stretch-wrapped, corner boards for glass and appliances.
- Labels: EAN-13 on unit, SSCC on pallet, "Fragile" for glass lids and tumblers.
""",
        )


def main() -> None:
    for old in OUT.rglob("*.md"):
        old.unlink()
    product_docs()
    raw_docs()
    supplier_docs()
    site_docs()
    customer_docs()
    minutes()
    incidents()
    staff_faqs()
    howtos()
    training()
    category_docs()
    print(f"generated {len(list(OUT.rglob('*.md')))} documents in {OUT}")


if __name__ == "__main__":
    main()
