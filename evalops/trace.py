"""Per-query observability traces.

One trace per query, emitted as a JSON line. A trace is the unit of
observability for a generative system (see corpus/observability.md): it records
what went in and what came out of a single query — latency, estimated token
cost, retrieval scores, and the two reliability signals that matter most,
whether the confidence gate tripped and whether the system refused.

JSONL is deliberate: the file appends one self-contained record per line, so a
run can be streamed, diffed against another run, or shipped to a monitoring
backend without a bespoke parser.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .system import Result

# A nominal blended price so the trace carries a cost signal, not just tokens.
# Order-of-magnitude only — the point is that cost is observable and trends, not
# that it matches a specific invoice.
USD_PER_1K_PROMPT_TOKENS = 0.003


@dataclass
class Trace:
    query_id: str
    query: str
    latency_ms: float
    prompt_tokens: int
    est_cost_usd: float
    top_score: float
    refused: bool
    gate_tripped: bool
    retrieved: list[dict]  # [{doc_id, chunk_id, score}, ...]
    citations: list[str]
    answer: str


def trace_from_result(result: Result) -> Trace:
    return Trace(
        query_id=result.query_id,
        query=result.query,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        est_cost_usd=round(result.prompt_tokens / 1000 * USD_PER_1K_PROMPT_TOKENS, 6),
        top_score=result.top_score,
        refused=result.refused,
        gate_tripped=result.gate_tripped,
        retrieved=[
            {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "score": h.score}
            for h in result.hits
        ],
        citations=result.citations,
        answer=result.answer,
    )


def write_traces(results: list[Result], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(trace_from_result(r)), ensure_ascii=False) + "\n")
    return path
