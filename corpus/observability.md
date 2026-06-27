# Observability

A generative system you cannot observe is a system you cannot operate. Unlike a
deterministic service, an LLM pipeline can degrade silently: retrieval drifts,
prompts regress, costs creep, and nothing throws an exception. Observability
makes that behaviour visible instead of guessed at.

The unit of observability is the per-query trace. Each trace records the inputs
and the outputs of a single query: latency, estimated token cost, the retrieval
scores and the documents retrieved, whether the confidence gate tripped, and
whether the system refused. Emitting traces as structured JSON lines means they
can be aggregated, diffed between runs, and shipped to a monitoring backend
without bespoke parsing.

Traces serve two audiences. In development they explain why a metric moved — a
recall drop usually shows up as lower retrieval scores on specific queries. In
production they are the raw material for monitoring: latency and cost budgets,
gate-trip rates, and refusal rates become dashboards and alerts.
