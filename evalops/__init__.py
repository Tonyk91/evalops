"""EvalOps — CI-gated evaluation and observability for LLM/RAG systems."""

from .system import RAGSystem, Result
from .metrics import aggregate, gate_classification, groundedness
from .gate import evaluate as gate_evaluate

__all__ = ["RAGSystem", "Result", "aggregate", "gate_classification",
           "groundedness", "gate_evaluate"]
