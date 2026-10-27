"""Score one agent run against one eval case. Pure functions, no I/O.

A case passes only if every check that applies to it passes. Checks:
  route      classifier picked the expected route
  status     completed vs awaiting_approval
  facts      every must_include item appears (an item may be a list of alternatives)
  forbidden  no must_not_include text appears
  tools      every expected tool was called
  no_writes  no proposal was queued unless the case expects one
  citation   for document questions, at least one cited source is an expected document
  proposal   the proposal used the expected tool and arguments
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

CASES_FILE = Path(__file__).parent / "cases.yaml"


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    cases = yaml.safe_load(path.read_text())
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    return cases


def _norm(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,450 -> 1450
    return re.sub(r"\s+", " ", text)


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(_subset(v, actual.get(k)) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and all(any(_subset(e, a) for a in actual) for e in expected)
    if isinstance(expected, str) and isinstance(actual, str):
        return _norm(expected) == _norm(actual)
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return abs(expected - actual) < 1e-6
    return expected == actual


def score(case: dict, result: dict) -> dict:
    checks: dict[str, bool] = {}
    answer = result.get("answer") or ""
    tools = result.get("tools_called") or []
    tool_log = result.get("tool_log") or []

    if "route" in case:
        checks["route"] = result.get("route") == case["route"]
    checks["status"] = result.get("status") == case.get("status", "completed")
    if case.get("must_include"):
        checks["facts"] = all(
            any(_contains(answer, alt) for alt in (item if isinstance(item, list) else [item]))
            for item in case["must_include"]
        )
    if case.get("must_not_include"):
        checks["forbidden"] = not any(_contains(answer, t) for t in case["must_not_include"])
    if case.get("tools"):
        checks["tools"] = all(t in tools for t in case["tools"])
    proposal = case.get("proposal")
    if not proposal:
        # A proposal the bridge refused (is_error) queued nothing, so it is not a write.
        checks["no_writes"] = not any(t["tool"].startswith("propose_") and not t.get("is_error") for t in tool_log)
    else:
        calls = [t for t in tool_log if t["tool"] == proposal["tool"] and not t.get("is_error")]
        checks["proposal"] = any(_subset(proposal.get("args", {}), c.get("args") or {}) for c in calls)
    if case.get("docs") and case.get("category") == "knowledge":
        cited = {c["doc_id"] for c in result.get("citations") or []}
        checks["citation"] = bool(cited & set(case["docs"]))
    return {"id": case["id"], "category": case["category"], "passed": all(checks.values()), "checks": checks}
