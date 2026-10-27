from evals.scoring import load_cases, score


def result(**kw):
    base = {
        "status": "completed",
        "route": "knowledge",
        "answer": "",
        "tools_called": [],
        "tool_log": [],
        "citations": [],
    }
    return {**base, **kw}


def test_case_file_is_valid():
    cases = load_cases()
    assert len(cases) == 100
    for c in cases:
        assert c["category"] in {"knowledge", "erp_data", "action", "permission", "out_of_scope"}, c["id"]


def test_facts_alternatives_and_number_formats():
    case = {"id": "x", "category": "erp_data", "must_include": [["1,240", "1240"], "KLCC"]}
    assert score(case, result(answer="Available: 1240 kg at klcc"))["passed"]
    assert not score(case, result(answer="Available: 1240 kg"))["passed"]


def test_citation_required_for_knowledge():
    case = {"id": "x", "category": "knowledge", "docs": ["POL-FIN-003"], "must_include": ["60 days"]}
    cited = result(answer="60 days [S1]", citations=[{"doc_id": "POL-FIN-003"}])
    assert score(case, cited)["passed"]
    assert not score(case, result(answer="60 days"))["checks"]["citation"]


def test_refused_proposal_is_not_a_write():
    case = {"id": "x", "category": "action"}
    refused = result(
        tools_called=["propose_stock_transfer"],
        tool_log=[{"tool": "propose_stock_transfer", "args": {}, "is_error": True}],
    )
    assert score(case, refused)["checks"]["no_writes"]
    queued = result(tool_log=[{"tool": "propose_stock_transfer", "args": {}, "is_error": False}])
    assert not score(case, queued)["checks"]["no_writes"]


def test_proposal_argument_subset():
    case = {
        "id": "x",
        "category": "action",
        "status": "awaiting_approval",
        "proposal": {
            "tool": "propose_stock_transfer",
            "args": {"to_warehouse": "KLCC", "lines": [{"sku": "TH-SA-6104"}]},
        },
    }
    run = result(
        status="awaiting_approval",
        tool_log=[
            {
                "tool": "propose_stock_transfer",
                "is_error": False,
                "args": {
                    "from_warehouse": "DC",
                    "to_warehouse": "klcc",
                    "lines": [{"sku": "TH-SA-6104", "quantity": 38}],
                },
            }
        ],
    )
    assert score(case, run)["passed"]
