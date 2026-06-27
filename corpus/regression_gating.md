# Regression gating in CI

Evaluation only protects quality if it runs automatically. A regression gate
wires the evaluation suite into continuous integration: on every push the suite
runs, the metrics are computed, and they are compared against a baseline that is
committed to the repository. If a metric drops below its baseline by more than a
small tolerance, the build fails.

This turns reliability into something enforced rather than hoped for. A change
that quietly worsens retrieval or weakens the gate cannot merge, because the
pipeline goes red. The baseline is version-controlled, so improving the system
is a deliberate act: you update the baseline in the same commit that improves the
metrics, and the diff records the gain.

A tolerance absorbs trivial noise without hiding real regressions. Metrics where
higher is better — hit rate, recall, groundedness, gate accuracy — are gated on
the downside. The gate's exit code is what CI reads: zero for pass, non-zero for
a regression, so no human has to remember to look at the numbers.
