"""A small, dependency-light TF-IDF retriever.

Hand-rolled on numpy rather than pulled from a library so the retrieval maths is
visible and the CI install stays tiny. EvalOps treats retrieval as a swappable
component; this is the deterministic, key-free default used in the bundled suite.
For dense-embedding retrieval (multilingual-e5, cross-encoder reranking) see the
companion projects InspectIQ and StaffFlow.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# A small English stop list. Without it, retrieval scores on function words alone,
# so an out-of-corpus question like "what is the capital of France?" matches on
# "what/is/the/of" and slips past the confidence gate. Dropping them means scores
# reflect content overlap, which is what the gate threshold is meant to read.
_STOPWORDS = frozenset(
    "a an and are as at be but by do does for from how if in into is it its of on or "
    "that the their them then there these they this to was what when where which who "
    "why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    text: str


class TfidfRetriever:
    """Cosine-similarity retrieval over a fixed set of chunks.

    The index is built once at construction; queries are O(vocab) per chunk.
    Scores are cosine similarities in [0, 1], which keeps the gate threshold
    interpretable.
    """

    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("TfidfRetriever needs at least one chunk")
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.doc_ids = [c["doc_id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]

        tokenized = [tokenize(t) for t in self.texts]
        self.vocab = {term: i for i, term in enumerate(sorted({t for toks in tokenized for t in toks}))}
        self.idf = self._compute_idf(tokenized)
        self.matrix = self._vectorize(tokenized)  # (n_chunks, vocab) L2-normalised

    def _compute_idf(self, tokenized: list[list[str]]) -> np.ndarray:
        n_docs = len(tokenized)
        df = np.zeros(len(self.vocab))
        for toks in tokenized:
            for term in set(toks):
                df[self.vocab[term]] += 1
        # smoothed idf, matches sklearn's default formulation
        return np.log((1 + n_docs) / (1 + df)) + 1.0

    def _vectorize(self, tokenized: list[list[str]]) -> np.ndarray:
        rows = np.zeros((len(tokenized), len(self.vocab)))
        for i, toks in enumerate(tokenized):
            for term in toks:
                rows[i, self.vocab[term]] += 1.0
            rows[i] *= self.idf
        return self._l2_normalize(rows)

    def _embed_query(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for term in tokenize(query):
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] += 1.0
        vec *= self.idf
        return self._l2_normalize(vec[None, :])[0]

    @staticmethod
    def _l2_normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        scores = self.matrix @ self._embed_query(query)
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(self.chunk_ids[i], self.doc_ids[i], round(float(scores[i]), 4), self.texts[i])
            for i in order
        ]
