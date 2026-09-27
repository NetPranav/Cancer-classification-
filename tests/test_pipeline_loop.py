import json

import numpy as np
import pytest

from oncopattern.cli import main
from oncopattern.data.catalogue import CATALOGUE, search
from oncopattern.loop.failures import CaseRecord, FailureLedger, fit_learning_curve, samples_needed, wilson_interval
from oncopattern.pipeline import COMPONENTS, DISCLAIMER, Config, OncoPattern
from oncopattern.reasoning.report import html_report


def test_phase_order_is_enforced():
    m = OncoPattern(Config())
    with pytest.raises(RuntimeError):
        m.fit([np.zeros((64, 64), np.float32)], ["normal"])
    with pytest.raises(ValueError):
        m.fit_normal([np.zeros((32, 32), np.float32)])


def test_calibration_output(trained):
    m, info, _ = trained
    assert set(info["weights"]) == set(COMPONENTS)
    assert info["weights"]["concept_evidence"] > 0  # faithfulness constraint
    assert info["n_needed_for_certificate"] == 45


def test_analysis_is_complete_and_explained(trained):
    m, _, test = trained
    a = m.analyze(test[0].image, "t0")
    assert abs(sum(a.probabilities.values()) - 1) < 1e-6
    assert a.prediction in a.prediction_set
    assert 1 <= a.tokens_read <= a.tokens_total == 64
    assert {e.verdict for e in a.evidence} <= {"supporting", "opposing", "uninformative"}
    assert len(a.evidence) == 8
    assert a.highlight.shape == (64, 64)
    assert "FINDING" in a.narrative and "LIMITS" in a.narrative
    json.dumps(a.to_dict(), default=float)
    page = html_report(a, test[0].image, DISCLAIMER, truth=test[0].label)
    assert page.startswith("<!doctype html>") and "data:image/png;base64," in page


def test_evaluation_beats_chance_and_localises(trained):
    m, _, test = trained
    metrics, ledger, _ = m.evaluate([p.image for p in test], [p.label for p in test],
                                    [p.meta for p in test], [p.mask for p in test])
    assert metrics["accuracy_all"] > 0.6  # chance is 0.25
    assert metrics["auroc_abnormal_vs_normal"] > 0.85
    assert metrics["pixel_auroc"] > 0.8
    assert ledger.summary()["cases"] == len(test)


def test_save_load_roundtrip(trained, tmp_path):
    m, _, test = trained
    m.save(str(tmp_path / "m.pt"))
    m2 = OncoPattern.load(str(tmp_path / "m.pt"))
    a, b = m.analyze(test[1].image), m2.analyze(test[1].image)
    assert a.probabilities == pytest.approx(b.probabilities)


def test_failure_ledger_slices_and_recommendations(tmp_path):
    led = FailureLedger()
    for i in range(40):
        bad = i % 2 == 0 and i < 30
        led.add(CaseRecord(f"c{i}", "nuclear_atypia" if i < 30 else "normal",
                           "normal" if bad else ("nuclear_atypia" if i < 30 else "normal"), 0.9, True, not bad,
                           {"modality": "histology", "organ": "breast", "site": "A" if i < 30 else "B"},
                           {"x": float(i)}, [float(i), float(bad)]))
    sl = {(s["key"], s["value"]): s for s in led.slices()}
    assert sl[("true_label", "nuclear_atypia")]["failure_rate"] == pytest.approx(0.5)
    recs = led.recommend()
    assert recs and any("PanNuke" in r["acquire"] or "BreakHis" in r["acquire"] for r in recs)
    assert led.clusters(2)
    led.save_regression_suite(str(tmp_path / "suite.json"))
    assert len(json.loads((tmp_path / "suite.json").read_text())) == 15


def test_learning_curve_and_wilson():
    ns = np.array([50, 100, 200, 400, 800])
    errs = 2.0 * ns ** -0.5 + 0.02
    a, b, c = fit_learning_curve(ns, errs)
    assert b == pytest.approx(0.5, abs=0.1) and c == pytest.approx(0.02, abs=0.01)
    assert samples_needed(a, b, c, 0.04) > 800
    assert samples_needed(a, b, c, 0.01) is None  # below the irreducible floor
    lo, hi = wilson_interval(0, 50)
    assert lo == 0 and 0 < hi < 0.1


def test_catalogue():
    assert len(CATALOGUE) >= 30
    assert any(d.name == "NLST" for d in search(organ="lung", longitudinal=True))
    assert all(d.domain == "pathology" for d in search(modality="H&E"))


def test_cli_catalogue(capsys):
    main(["catalogue", "--organ", "brain", "--longitudinal"])
    assert "RIDER Neuro MRI" in capsys.readouterr().out
