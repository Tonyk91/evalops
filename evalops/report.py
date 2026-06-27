"""Run artifacts: a metrics badge, a markdown summary, and an HTML report.

These are what a human or a CI dashboard looks at after a run. The badge is a
shields.io endpoint JSON so a README can render a live pass/fail badge; the
markdown summary is dropped into the GitHub Actions job summary; the HTML report
is uploaded as a build artifact for the per-query detail.
"""

from __future__ import annotations

import json
from pathlib import Path

from .gate import Verdict


def write_badge(verdict: Verdict, path: str | Path) -> Path:
    """shields.io 'endpoint' schema — README points a badge at this file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "label": "evalops",
        "message": "passing" if verdict.passed else "regressed",
        "color": "brightgreen" if verdict.passed else "red",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_markdown_summary(verdict: Verdict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## EvalOps — {'✅ PASS' if verdict.passed else '❌ FAIL'}",
        "",
        f"Regression gate, tolerance ±{verdict.tolerance}.",
        "",
        "| Metric | Current | Baseline | Δ | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for c in sorted(verdict.checks, key=lambda c: c.name):
        status = "ok" if c.ok else "**REGRESSION**"
        lines.append(
            f"| {c.name} | {c.current:.4f} | {c.baseline:.4f} | {c.delta:+.4f} | {status} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_html_report(verdict: Verdict, traces_path: str | Path,
                      path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    traces = _read_traces(traces_path)

    rows = "\n".join(
        f"<tr class='{'bad' if not c.ok else ''}'>"
        f"<td>{c.name}</td><td>{c.current:.4f}</td><td>{c.baseline:.4f}</td>"
        f"<td>{c.delta:+.4f}</td><td>{'ok' if c.ok else 'REGRESSION'}</td></tr>"
        for c in sorted(verdict.checks, key=lambda c: c.name)
    )
    trace_rows = "\n".join(
        f"<tr class='{'refused' if t['refused'] else ''}'>"
        f"<td>{t['query_id']}</td><td>{t['query']}</td>"
        f"<td>{t['top_score']:.4f}</td><td>{'yes' if t['refused'] else 'no'}</td>"
        f"<td>{t['latency_ms']:.2f}</td><td>${t['est_cost_usd']:.6f}</td>"
        f"<td>{', '.join(h['doc_id'] for h in t['retrieved'])}</td></tr>"
        for t in traces
    )
    verdict_class = "pass" if verdict.passed else "fail"
    verdict_text = "PASS" if verdict.passed else "FAIL — regression"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>EvalOps report</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;color:#1a1a2e}}
 h1{{margin-bottom:.2rem}} .verdict{{font-weight:700;padding:.3rem .7rem;border-radius:6px;color:#fff;display:inline-block}}
 .pass{{background:#16a34a}} .fail{{background:#dc2626}}
 table{{border-collapse:collapse;margin:1rem 0;width:100%}}
 th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}}
 th{{background:#f3f3fb}} tr.bad td{{background:#fde8e8}} tr.refused td{{color:#888}}
 td:nth-child(n+3){{font-variant-numeric:tabular-nums}}
</style></head><body>
<h1>EvalOps report</h1>
<p><span class="verdict {verdict_class}">{verdict_text}</span>
 &nbsp;regression gate, tolerance ±{verdict.tolerance}</p>
<h2>Metrics vs baseline</h2>
<table><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Δ</th><th>Status</th></tr>
{rows}</table>
<h2>Per-query traces</h2>
<table><tr><th>Query id</th><th>Query</th><th>Top score</th><th>Refused</th>
<th>Latency (ms)</th><th>Est cost</th><th>Retrieved docs</th></tr>
{trace_rows}</table>
</body></html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def _read_traces(traces_path: str | Path) -> list[dict]:
    text = Path(traces_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
