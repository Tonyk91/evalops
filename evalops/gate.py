"""The regression gate.

Compares this run's metrics against a committed baseline and decides whether the
build passes. Every bundled metric is "higher is better", so each is gated on the
downside: a metric may not fall more than `tolerance` below its baseline. The
tolerance absorbs trivial noise without hiding a real regression.

`evaluate` returns a structured verdict; the CLI turns `verdict.passed` into the
process exit code so CI reads the result without a human in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TOLERANCE = 0.02


@dataclass
class MetricCheck:
    name: str
    current: float
    baseline: float
    delta: float
    ok: bool


@dataclass
class Verdict:
    passed: bool
    checks: list[MetricCheck]
    tolerance: float

    @property
    def regressions(self) -> list[MetricCheck]:
        return [c for c in self.checks if not c.ok]


def evaluate(current: dict[str, float], baseline: dict[str, float],
             tolerance: float = DEFAULT_TOLERANCE) -> Verdict:
    checks: list[MetricCheck] = []
    for name, base in baseline.items():
        cur = current.get(name)
        if cur is None:
            # A metric present in the baseline but missing now is itself a regression.
            checks.append(MetricCheck(name, float("nan"), base, float("nan"), ok=False))
            continue
        delta = round(cur - base, 4)
        checks.append(MetricCheck(name, cur, base, delta, ok=delta >= -tolerance))
    return Verdict(passed=all(c.ok for c in checks), checks=checks, tolerance=tolerance)


def format_report(verdict: Verdict) -> str:
    lines = [
        f"{'metric':<22} {'current':>9} {'baseline':>9} {'delta':>8}  status",
        "-" * 62,
    ]
    for c in sorted(verdict.checks, key=lambda c: c.name):
        status = "ok" if c.ok else "REGRESSION"
        lines.append(
            f"{c.name:<22} {c.current:>9.4f} {c.baseline:>9.4f} {c.delta:>+8.4f}  {status}"
        )
    lines.append("-" * 62)
    lines.append("PASS" if verdict.passed else "FAIL — metrics regressed beyond tolerance")
    return "\n".join(lines)
