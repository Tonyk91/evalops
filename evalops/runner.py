"""Orchestration and CLI.

Wires the pieces into one command: load a suite, run every case through the RAG
system under test, compute metrics, emit observability traces and reports, then
gate the metrics against a committed baseline and exit non-zero on a regression.
This module is what CI invokes (`python -m evalops eval`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import gate as gate_mod
from . import report as report_mod
from .metrics import aggregate
from .system import RAGSystem
from .trace import write_traces

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = ROOT / "suites" / "default.yaml"
DEFAULT_CORPUS = ROOT / "corpus"
DEFAULT_BASELINE = ROOT / "suites" / "baseline.json"
REPORTS_DIR = ROOT / "reports"


def load_suite(path: str | Path) -> dict:
    suite = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not suite.get("cases"):
        raise ValueError(f"suite {path} has no cases")
    return suite


def run_suite(suite: dict, corpus_dir: str | Path):
    system = RAGSystem(
        corpus_dir=str(corpus_dir),
        top_k=suite.get("top_k", 4),
        gate_threshold=suite.get("gate_threshold", 0.12),
    )
    results = []
    gold: dict[str, dict] = {}
    for case in suite["cases"]:
        results.append(system.answer(case["id"], case["query"]))
        gold[case["id"]] = {
            "relevant_docs": case.get("relevant_docs", []),
            "expect_refusal": case.get("expect_refusal", False),
        }
    metrics = aggregate(results, gold, k=suite.get("top_k", 4))
    return results, metrics


def cmd_eval(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    results, metrics = run_suite(suite, args.corpus)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    traces_path = write_traces(results, REPORTS_DIR / "traces.jsonl")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name:<22} {value:.4f}")
    print()

    if args.update_baseline:
        Path(args.baseline).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline updated → {args.baseline}")
        return 0

    if not Path(args.baseline).exists():
        print(f"No baseline at {args.baseline}. Create one with --update-baseline.",
              file=sys.stderr)
        return 2

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    verdict = gate_mod.evaluate(metrics, baseline, tolerance=args.tolerance)

    report_mod.write_badge(verdict, REPORTS_DIR / "badge.json")
    report_mod.write_markdown_summary(verdict, REPORTS_DIR / "summary.md")
    report_mod.write_html_report(verdict, traces_path, REPORTS_DIR / "report.html")

    print(gate_mod.format_report(verdict))
    _append_github_summary(REPORTS_DIR / "summary.md")
    return 0 if verdict.passed else 1


def _append_github_summary(summary_md: Path) -> None:
    import os

    target = os.getenv("GITHUB_STEP_SUMMARY")
    if target and summary_md.exists():
        with open(target, "a", encoding="utf-8") as f:
            f.write(summary_md.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalops", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="run the suite and gate against the baseline")
    ev.add_argument("--suite", default=str(DEFAULT_SUITE))
    ev.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ev.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ev.add_argument("--tolerance", type=float, default=gate_mod.DEFAULT_TOLERANCE)
    ev.add_argument("--update-baseline", action="store_true",
                    help="write current metrics as the new baseline instead of gating")
    ev.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
