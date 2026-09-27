import numpy as np
import pytest

from oncopattern.calibration.conformal import (ConformalClassifier, SelectiveRiskController, fit_temperature,
                                               min_samples_for_guarantee, softmax)
from oncopattern.longitudinal.change import cusum, doubling_time, kalman_trend, longitudinal_report
from oncopattern.reasoning.evidence import ConceptEvidenceModel


def _synthetic(n, rng, K=3, sharp=2.0):
    y = rng.integers(0, K, n)
    logits = rng.normal(0, 1, (n, K))
    logits[np.arange(n), y] += sharp
    return logits, y


def test_evidence_decomposes_log_odds_exactly():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (20, 3)), rng.normal([3, 0, 0], 1, (20, 3))])
    y = np.repeat([0, 1], 20)
    m = ConceptEvidenceModel(["a", "b", "c"]).fit(X, y, 2)
    x = np.array([[3.0, 0.1, -0.2]])
    items = m.explain(x, 1, 0)
    total_db = sum(i.woe_db for i in items) + m.prior_db(1, 0)
    lg = m.logits(x)[0]
    assert total_db == pytest.approx(10 * (lg[1] - lg[0]) / np.log(10), rel=1e-6)
    assert items[0].concept == "a" and items[0].verdict == "supporting"


def test_temperature_fixes_overconfidence():
    rng = np.random.default_rng(0)
    logits, y = _synthetic(2000, rng)
    assert fit_temperature(logits * 5, y) > 2.5


def test_conformal_coverage_holds():
    rng = np.random.default_rng(1)
    cal_l, cal_y = _synthetic(500, rng)
    te_l, te_y = _synthetic(4000, rng)
    cp = ConformalClassifier(alpha=0.1).fit(softmax(cal_l), cal_y)
    sets = cp.predict_set(softmax(te_l))
    cov = sets[np.arange(len(te_y)), te_y].mean()
    assert 0.87 <= cov <= 0.95


def test_selective_risk_control_bounds_error_on_answered():
    rng = np.random.default_rng(2)
    cal_l, cal_y = _synthetic(3000, rng, sharp=2.5)
    te_l, te_y = _synthetic(20000, rng, sharp=2.5)
    sel = SelectiveRiskController(target_risk=0.05, delta=0.1).fit(softmax(cal_l), cal_y)
    assert sel.result.certified
    p = softmax(te_l)
    acc = sel.accept(p)
    err = (p[acc].argmax(1) != te_y[acc]).mean()
    assert err <= 0.06 and 0 < acc.mean() < 1


def test_selective_refuses_to_certify_with_too_little_data():
    rng = np.random.default_rng(3)
    l, y = _synthetic(30, rng)
    sel = SelectiveRiskController(0.01, 0.1).fit(softmax(l), y)
    assert not sel.result.certified and not sel.accept(softmax(l)).any()
    assert min_samples_for_guarantee(0.01, 0.1) == 230


def test_longitudinal_detects_growth():
    t = np.arange(6) * 6.0
    flat = kalman_trend(t, [1, 1.1, 0.9, 1.0, 1.05, 0.95])
    grow = kalman_trend(t, [1, 1, 1.5, 2.5, 4, 6])
    assert abs(flat[-1, 1]) < 0.02 and grow[-1, 1] > 0.1
    path, alarm = cusum([0, 0.1, -0.1, 3, 4, 5], 0, 1)  # S = 0, 0, 0, 2.5, 6.0 > h=4
    assert alarm == 4 and path[3] == pytest.approx(2.5)
    assert doubling_time([0, 1, 2], [1, 2, 4]) == pytest.approx(1.0)
    rep = longitudinal_report(t, [1, 1, 1.5, 2.5, 4, 6], 1.0, 0.1)
    assert rep["trend_significant"] and rep["change_detected_at"] is not None
