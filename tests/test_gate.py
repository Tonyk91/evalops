"""Unit tests for the regression gate decision logic."""

from __future__ import annotations

import math

from evalops.gate import evaluate


def test_pass_when_metrics_hold():
    baseline = {"recall@4": 0.95, "groundedness": 0.80}
    current = {"recall@4": 0.96, "groundedness": 0.80}
    verdict = evaluate(current, baseline, tolerance=0.02)
    assert verdict.passed
    assert verdict.regressions == []


def test_small_dip_within_tolerance_passes():
    baseline = {"recall@4": 0.95}
    current = {"recall@4": 0.94}  # -0.01, within ±0.02
    assert evaluate(current, baseline, tolerance=0.02).passed


def test_regression_beyond_tolerance_fails():
    baseline = {"recall@4": 0.95}
    current = {"recall@4": 0.90}  # -0.05
    verdict = evaluate(current, baseline, tolerance=0.02)
    assert not verdict.passed
    assert [c.name for c in verdict.regressions] == ["recall@4"]


def test_missing_metric_is_a_regression():
    verdict = evaluate({}, {"recall@4": 0.95})
    assert not verdict.passed
    assert math.isnan(verdict.checks[0].current)
