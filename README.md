# EvalOps

[![evalops](https://github.com/Tonyk91/evalops/actions/workflows/evalops.yml/badge.svg)](https://github.com/Tonyk91/evalops/actions/workflows/evalops.yml)

A reliability layer for LLM/RAG systems: an evaluation suite that runs in CI,
**fails the build when quality regresses** against a committed baseline, and emits
structured observability traces for every query.

The half of "production AI" most demos skip is the part that proves the system
keeps working. EvalOps is that part, running green and reproducible.

**Stack:** Python · GitHub Actions · numpy · scikit-learn

## What it does

- **Evaluation suite.** A YAML suite of queries with gold document labels and
  deliberate out-of-corpus questions, run against a pluggable RAG system. Computes
  retrieval metrics (hit rate, MRR, recall@k), answer groundedness, and the
  confidence-gate's precision/recall/F1 (scikit-learn) — refusing when evidence is
  weak is scored as a first-class behaviour, not a side effect.
- **Regression gate in CI.** On every push the suite runs and each metric is
  compared against `suites/baseline.json`. If any drops more than the tolerance
  below baseline, the gate exits non-zero and the build goes red. Quality is
  enforced, not hoped for. Improving the system means updating the baseline in the
  same commit — the diff records the gain.
- **Observability traces.** One JSON line per query in `reports/traces.jsonl`:
  latency, estimated token cost, retrieval scores and retrieved docs, and the two
  reliability signals — whether the confidence gate tripped and whether the system
  refused. The raw material for monitoring a generative system instead of guessing.
- **Reports.** An HTML report (`reports/report.html`), a markdown job summary
  surfaced in the GitHub Actions run, and a shields-endpoint badge.

## Why it's built this way

- **Key-free and deterministic.** The bundled system-under-test uses a hand-rolled
  TF-IDF retriever and extractive generation, so a run is byte-for-byte
  reproducible with no model download and no API key — CI is always green for the
  right reason. Set `EVALOPS_LLM=1` with `ANTHROPIC_API_KEY` to swap in
  Claude-generated answers; the harness is identical either way.
- **The system is a black box.** Anything exposing `.answer(query) -> Result` can
  be measured. The bundled pipeline is a reference; the metrics, gate and traces
  are the product. (For dense-embedding retrieval with reranking, see the companion
  projects InspectIQ and StaffFlow.)
- **The gate is the trust boundary.** AI proposes, a human approves: below the
  confidence threshold the system refuses and routes to a human rather than
  answering on weak evidence — and how well it does that is itself measured.

## Run it

```bash
pip install -r requirements.txt

python -m evalops eval                 # run suite + gate against baseline (CI does this)
python -m evalops eval --update-baseline   # accept current metrics as the new baseline
pytest -q                              # unit tests for metrics + gate logic
```

A regression looks like this — the build fails and says exactly which metric moved:

```
metric                   current  baseline    delta  status
--------------------------------------------------------------
recall@4                  0.3333    1.0000  -0.6667  REGRESSION
--------------------------------------------------------------
FAIL — metrics regressed beyond tolerance      (exit code 1)
```

## Layout

```
evalops/
  corpus.py       load a markdown corpus into citable chunks
  retriever.py    hand-rolled TF-IDF retriever (numpy)
  system.py       the RAG system under test: retrieve → gate → ground
  metrics.py      retrieval + answer + gate metrics (scikit-learn)
  trace.py        per-query observability traces (JSONL)
  gate.py         regression comparison vs baseline → pass/fail
  report.py       badge + markdown summary + HTML report
  runner.py       orchestration + CLI (python -m evalops)
corpus/           the knowledge base (one .md per document)
suites/           default.yaml suite + committed baseline.json
tests/            unit tests for metrics and the gate
.github/workflows/evalops.yml   the CI reliability gate
```
