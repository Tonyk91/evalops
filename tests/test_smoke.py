"""End-to-end smoke test: the bundled suite runs and refusal cases refuse."""

from __future__ import annotations

from pathlib import Path

from evalops.runner import DEFAULT_CORPUS, DEFAULT_SUITE, load_suite, run_suite

ROOT = Path(__file__).resolve().parent.parent


def test_suite_runs_end_to_end():
    suite = load_suite(DEFAULT_SUITE)
    results, metrics = run_suite(suite, DEFAULT_CORPUS)
    assert len(results) == len(suite["cases"])
    # core metrics are well-formed probabilities
    for key in ["recall@4", "gate_accuracy", "groundedness"]:
        assert 0.0 <= metrics[key] <= 1.0


def test_out_of_corpus_questions_are_refused():
    suite = load_suite(DEFAULT_SUITE)
    results, _ = run_suite(suite, DEFAULT_CORPUS)
    by_id = {r.query_id: r for r in results}
    assert by_id["refuse-capital"].refused
    assert by_id["refuse-vacation"].refused
