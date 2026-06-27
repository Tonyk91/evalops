"""Unit tests for the metric functions — pure functions of (results, gold)."""

from __future__ import annotations

from evalops.metrics import (
    aggregate,
    gate_classification,
    groundedness,
    hit_rate,
    recall_at_k,
    reciprocal_rank,
)
from evalops.retriever import Hit
from evalops.system import Result


def make_result(query_id, hits, answer="", citations=None, refused=False):
    return Result(
        query_id=query_id, query="q", hits=hits, answer=answer,
        citations=citations or [], refused=refused, gate_tripped=refused,
        top_score=hits[0].score if hits else 0.0, latency_ms=1.0,
    )


def test_hit_rate_and_mrr_rank():
    hits = [Hit("b#0", "b", 0.5, "x"), Hit("a#0", "a", 0.4, "y")]
    r = make_result("q1", hits)
    assert hit_rate(r, ["a"]) == 1.0
    assert hit_rate(r, ["c"]) == 0.0
    assert reciprocal_rank(r, ["a"]) == 0.5  # second position
    assert reciprocal_rank(r, ["b"]) == 1.0  # first position


def test_recall_at_k_partial():
    hits = [Hit("a#0", "a", 0.5, "x"), Hit("b#0", "b", 0.4, "y")]
    r = make_result("q1", hits)
    assert recall_at_k(r, ["a", "b"]) == 1.0
    assert recall_at_k(r, ["a", "c"]) == 0.5


def test_groundedness_rewards_cited_overlap():
    hits = [Hit("a#0", "a", 0.5, "the sky is blue today")]
    grounded = make_result("q1", hits, answer="the sky is blue", citations=["a#0"])
    drifted = make_result("q2", hits, answer="bananas grow in tropics", citations=["a#0"])
    assert groundedness(grounded) == 1.0
    assert groundedness(drifted) < 0.3


def test_refusal_is_vacuously_grounded():
    r = make_result("q1", [Hit("a#0", "a", 0.0, "x")], refused=True)
    assert groundedness(r) == 1.0


def test_gate_classification_perfect():
    gold = {"a": {"expect_refusal": False}, "b": {"expect_refusal": True}}
    results = [
        make_result("a", [Hit("a#0", "a", 0.5, "x")], refused=False),
        make_result("b", [Hit("z#0", "z", 0.0, "x")], refused=True),
    ]
    rep = gate_classification(results, gold)
    assert rep["gate_precision"] == 1.0
    assert rep["gate_recall"] == 1.0
    assert rep["gate_f1"] == 1.0


def test_gate_recall_drops_when_ungrounded_answered():
    # gate should have refused 'b' but answered it → recall on refusal falls
    gold = {"a": {"expect_refusal": False}, "b": {"expect_refusal": True}}
    results = [
        make_result("a", [Hit("a#0", "a", 0.5, "x")], refused=False),
        make_result("b", [Hit("z#0", "z", 0.5, "x")], refused=False),
    ]
    rep = gate_classification(results, gold)
    assert rep["gate_recall"] == 0.0


def test_aggregate_keys_present():
    gold = {"a": {"relevant_docs": ["a"], "expect_refusal": False}}
    results = [make_result("a", [Hit("a#0", "a", 0.5, "x")], answer="x", citations=["a#0"])]
    summary = aggregate(results, gold, k=4)
    for key in ["hit_rate@4", "mrr@4", "recall@4", "gate_accuracy",
                "groundedness", "answer_rate", "gate_precision", "gate_recall"]:
        assert key in summary
