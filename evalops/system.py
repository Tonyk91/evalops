"""The RAG system under test.

This is a deliberately small, deterministic reference implementation of the kind of
system EvalOps is built to evaluate: retrieve, gate on confidence, then ground the
answer in the retrieved context. The point of EvalOps is that the system is a
*pluggable black box* — anything exposing `.answer(query) -> Result` can be measured.

Two design choices keep the bundled pipeline reproducible in CI:

* Retrieval is TF-IDF (see `retriever.py`) — no model download, no API key.
* Generation is extractive — the answer is lifted verbatim from the cited chunk, so
  it is grounded by construction and the run is byte-for-byte deterministic. Set
  `EVALOPS_LLM=1` (with `ANTHROPIC_API_KEY`) to swap in a Claude-generated answer;
  the evaluation harness is identical either way.

The confidence gate is the trust boundary: when the top retrieval score is below
`gate_threshold`, the system refuses and routes to a human instead of answering on
weak evidence. Measuring how well that gate behaves is a first-class metric.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from .corpus import load_chunks
from .retriever import Hit, TfidfRetriever

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
REFUSAL = "Not enough grounded evidence to answer confidently — routing to human review."


@dataclass
class Result:
    query_id: str
    query: str
    hits: list[Hit]
    answer: str
    citations: list[str]
    refused: bool
    gate_tripped: bool
    top_score: float
    latency_ms: float
    prompt_tokens: int = 0
    extra: dict = field(default_factory=dict)


class RAGSystem:
    def __init__(self, corpus_dir: str, top_k: int = 4, gate_threshold: float = 0.12):
        self.top_k = top_k
        self.gate_threshold = gate_threshold
        self.retriever = TfidfRetriever(load_chunks(corpus_dir))
        self._use_llm = os.getenv("EVALOPS_LLM") == "1"

    def answer(self, query_id: str, query: str) -> Result:
        start = time.perf_counter()
        hits = self.retriever.search(query, self.top_k)
        top_score = hits[0].score if hits else 0.0
        # crude token estimate of what a real prompt would cost (context + query)
        prompt_tokens = self._estimate_tokens(query, hits)

        if top_score < self.gate_threshold:
            return Result(
                query_id=query_id, query=query, hits=hits, answer=REFUSAL,
                citations=[], refused=True, gate_tripped=True, top_score=top_score,
                latency_ms=self._elapsed(start), prompt_tokens=prompt_tokens,
            )

        answer, citations = self._generate(query, hits)
        return Result(
            query_id=query_id, query=query, hits=hits, answer=answer,
            citations=citations, refused=False, gate_tripped=False, top_score=top_score,
            latency_ms=self._elapsed(start), prompt_tokens=prompt_tokens,
        )

    def _generate(self, query: str, hits: list[Hit]) -> tuple[str, list[str]]:
        best = hits[0]
        if self._use_llm:
            return self._generate_llm(query, hits), [best.chunk_id]
        # extractive default: first two sentences of the top chunk, grounded by construction
        sentences = _SENTENCE.split(best.text.strip())
        answer = " ".join(sentences[:2]).strip()
        return answer, [best.chunk_id]

    def _generate_llm(self, query: str, hits: list[Hit]) -> str:  # pragma: no cover - needs key
        from anthropic import Anthropic

        context = "\n\n".join(f"[{h.chunk_id}] {h.text}" for h in hits)
        client = Anthropic()
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=300,
            system="Answer only from the provided context and cite the chunk ids you used. "
                   "If the context does not contain the answer, say so.",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
        )
        return msg.content[0].text.strip()

    @staticmethod
    def _estimate_tokens(query: str, hits: list[Hit]) -> int:
        words = len(query.split()) + sum(len(h.text.split()) for h in hits)
        return int(words * 1.3)  # ~1.3 tokens/word, English heuristic

    @staticmethod
    def _elapsed(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)
