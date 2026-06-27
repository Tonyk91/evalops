"""Evaluation metrics for a RAG system.

Two families:

* **Retrieval** — did the right document make it into the top-k? Computed at the
  document level: a hit is any retrieved chunk whose `doc_id` is in the gold set.
  hit_rate, mrr and recall are the standard signals for whether retrieval is the
  bottleneck (it usually is).
* **Answer / reliability** — does the system answer when it should and refuse when
  it shouldn't, and is the answer grounded in what it cited? `gate_accuracy` scores
  the trust boundary directly; `groundedness` checks the answer against its citation.

All metrics are pure functions of (results, gold) so they are trivial to unit-test
and to diff between runs in the regression gate.
"""

from __future__ import annotations

from .retriever import tokenize
from .system import Result


def _relevant(result: Result, gold_docs: list[str]) -> list[bool]:
    return [hit.doc_id in gold_docs for hit in result.hits]


def hit_rate(result: Result, gold_docs: list[str]) -> float:
    return 1.0 if any(_relevant(result, gold_docs)) else 0.0


def reciprocal_rank(result: Result, gold_docs: list[str]) -> float:
    for rank, is_rel in enumerate(_relevant(result, gold_docs), start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def recall_at_k(result: Result, gold_docs: list[str]) -> float:
    if not gold_docs:
        return 0.0
    retrieved_docs = {hit.doc_id for hit in result.hits}
    found = sum(1 for d in gold_docs if d in retrieved_docs)
    return found / len(gold_docs)


def groundedness(result: Result) -> float:
    """Fraction of answer tokens that also appear in the cited chunk(s).

    A blunt but honest proxy for faithfulness: an answer that drifts away from its
    citation scores low. Refusals are excluded (nothing was asserted).
    """
    if result.refused or not result.citations:
        return 1.0  # refusing asserts nothing, so it cannot be unfaithful
    cited_text = " ".join(h.text for h in result.hits if h.chunk_id in result.citations)
    cited_tokens = set(tokenize(cited_text))
    answer_tokens = tokenize(result.answer)
    if not answer_tokens:
        return 0.0
    grounded = sum(1 for t in answer_tokens if t in cited_tokens)
    return grounded / len(answer_tokens)


def gate_classification(results: list[Result], gold: dict[str, dict]) -> dict[str, float]:
    """Precision/recall/F1 of the confidence gate, treating "refuse" as the
    positive class.

    The gate is a binary classifier — refuse or answer — so the right tool to
    score it is a classification report, not a single accuracy number. Precision
    answers "when the gate refused, should it have?" (low precision = refusing
    answerable questions); recall answers "of the questions it should have
    refused, how many did it?" (low recall = answering ungrounded questions, the
    dangerous direction). Computed with scikit-learn so the numbers match any
    standard report.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = [bool(gold[r.query_id].get("expect_refusal")) for r in results]
    y_pred = [r.refused for r in results]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[True], average="binary", pos_label=True,
        zero_division=0.0,
    )
    return {
        "gate_precision": round(float(precision), 4),
        "gate_recall": round(float(recall), 4),
        "gate_f1": round(float(f1), 4),
    }


def aggregate(results: list[Result], gold: dict[str, dict], k: int) -> dict[str, float]:
    """Roll per-query metrics up into the scalar summary the gate compares.

    `gold[query_id]` carries `relevant_docs` and an `expect_refusal` flag. The gate
    metric `gate_accuracy` rewards refusing out-of-corpus questions and answering
    in-corpus ones.
    """
    n = len(results)
    answerable = [r for r in results if not gold[r.query_id].get("expect_refusal")]

    hit = mrr = rec = 0.0
    for r in answerable:
        docs = gold[r.query_id].get("relevant_docs", [])
        hit += hit_rate(r, docs)
        mrr += reciprocal_rank(r, docs)
        rec += recall_at_k(r, docs)
    n_ans = len(answerable) or 1

    gate_correct = sum(
        1 for r in results
        if r.refused == bool(gold[r.query_id].get("expect_refusal"))
    )
    grounded = sum(groundedness(r) for r in results) / (n or 1)

    summary = {
        f"hit_rate@{k}": round(hit / n_ans, 4),
        f"mrr@{k}": round(mrr / n_ans, 4),
        f"recall@{k}": round(rec / n_ans, 4),
        "gate_accuracy": round(gate_correct / (n or 1), 4),
        "groundedness": round(grounded, 4),
        "answer_rate": round(sum(1 for r in results if not r.refused) / (n or 1), 4),
    }
    summary.update(gate_classification(results, gold))
    return summary
