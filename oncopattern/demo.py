"""End-to-end demonstrations on phantoms (every phase, with ground truth)."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from oncopattern.data.phantoms import MRI_PATTERNS, longitudinal_mri, make_tissue_set, mri_slice, tissue_patch
from oncopattern.longitudinal.change import longitudinal_report
from oncopattern.loop.failures import fit_learning_curve, samples_needed
from oncopattern.pipeline import DISCLAIMER, Config, OncoPattern
from oncopattern.reasoning.report import html_report


def _log(msg):
    print(f"[oncopattern] {msg}", flush=True)


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float))


def run_tissue_demo(out: str | Path, quick: bool = False, seed: int = 0, n_reports: int = 6) -> dict:
    out = Path(out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    per = dict(unl=15 if quick else 40, normals=50 if quick else 80, support=8 if quick else 12,
               cal=30 if quick else 50, test=20 if quick else 40)
    cfg = Config(ssl_epochs=2 if quick else 4, duplex_epochs=25 if quick else 60, seed=seed)
    model = OncoPattern(cfg)

    _log("Phase 3  label-free pretraining (masked-view consistency, D4-equivariant encoder)")
    ssl = model.pretrain([p.image for p in make_tissue_set(rng, per["unl"])])
    _log("Phase 2  normality memory from normal tissue only")
    model.fit_normal([tissue_patch(rng).image for _ in range(per["normals"])])
    after_normal = copy.deepcopy(model)
    _log(f"Phase 4-5  few-shot fit: {per['support']} labelled examples per pattern")
    sup = make_tissue_set(rng, per["support"])
    model.fit([p.image for p in sup], [p.label for p in sup])
    _log("Phase 7  calibration: fusion, temperature, conformal sets, risk-controlled abstention")
    cal = make_tissue_set(rng, per["cal"])
    calib = model.calibrate([p.image for p in cal], [p.label for p in cal])
    _log("Phase 8  evaluation + failure ledger")
    test = make_tissue_set(rng, per["test"])
    ids = [f"tissue_{i:04d}" for i in range(len(test))]
    metrics, ledger, analyses = model.evaluate([p.image for p in test], [p.label for p in test],
                                               [p.meta for p in test], [p.mask for p in test], ids)

    _log("Phase 8  learning curve: how much labelled data would close the gap?")
    curve = learning_curve(after_normal, rng, cal, test, sizes=(3, 6, 12) if quick else (3, 6, 12, 24),
                           epochs=cfg.duplex_epochs // 2)

    # reports: the failures first (that is where we learn), then a few correct ones
    order = sorted(range(len(test)), key=lambda i: (analyses[i].prediction == test[i].label and analyses[i].answered, i))
    written = []
    for i in order[:n_reports]:
        path = out / "reports" / f"{ids[i]}.html"
        path.write_text(html_report(analyses[i], test[i].image, DISCLAIMER, truth=test[i].label))
        (out / "reports" / f"{ids[i]}.json").write_text(json.dumps(analyses[i].to_dict(), indent=2, default=float))
        written.append(str(path))
    failure_report = ledger.report()
    _write_json(out / "metrics.json", {"metrics": metrics, "calibration": calib,
                                       "ssl_loss_first_last": [ssl[0], ssl[-1]] if ssl else None,
                                       "seconds": time.time() - t0, "sizes": per})
    failure_report["learning_curve"] = curve
    _write_json(out / "failure_report.json", failure_report)
    ledger.save_regression_suite(str(out / "regression_suite.json"))
    model.save(str(out / "model.pt"))
    _log(f"done in {time.time() - t0:.0f}s -> {out}")
    return {"metrics": metrics, "calibration": calib, "failures": failure_report, "reports": written}


def learning_curve(base: OncoPattern, rng, cal, test, sizes=(3, 6, 12, 24), epochs: int = 30) -> dict:
    """Refit the few-shot heads at several support sizes (sharing the normal bank), then fit
    err(n) = a n^-b + c and solve for the n that reaches 5% and 1% error."""
    ns, errs = [], []
    for k in sizes:
        m = copy.deepcopy(base)
        m.cfg.duplex_epochs, m.cfg.duplex_augment = epochs, False
        sup = make_tissue_set(rng, k)
        m.fit([p.image for p in sup], [p.label for p in sup])
        m.calibrate([p.image for p in cal], [p.label for p in cal])
        met, _, _ = m.evaluate([p.image for p in test], [p.label for p in test])
        ns.append(k * len(m.classes))
        errs.append(1 - met["accuracy_all"])
        _log(f"  n={ns[-1]:>4} labelled -> error {errs[-1]:.3f}")
    a, b, c = fit_learning_curve(ns, np.maximum(errs, 1e-3))
    return {"n_labelled": ns, "error": errs, "fit": {"a": a, "b": b, "floor_c": c},
            "n_for_5pct": samples_needed(a, b, c, 0.05), "n_for_1pct": samples_needed(a, b, c, 0.01)}


def run_longitudinal_demo(out: str | Path, quick: bool = False, seed: int = 1) -> dict:
    """Head-MRI phantoms: learn normal anatomy, then follow one patient over five scans."""
    out = Path(out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = Config(classes=MRI_PATTERNS, ssl_epochs=1 if quick else 2, duplex_epochs=20 if quick else 40,
                 duplex_augment=False, seed=seed)
    model = OncoPattern(cfg)
    n = 30 if quick else 50
    model.fit_normal([mri_slice(rng).image for _ in range(n)])

    def labelled(k):
        items = [mri_slice(rng) for _ in range(k)] + [mri_slice(rng, lesion_radius=rng.uniform(1.2, 4.0)) for _ in range(k)]
        return [p.image for p in items], [p.label for p in items]

    model.fit(*labelled(10))
    model.calibrate(*labelled(50 if quick else 60))

    times = [0, 6, 12, 18, 24]
    series = longitudinal_mri(rng, radii=(0.0, 0.0, 1.2, 2.2, 3.4))
    analyses = model.analyze_batch([p.image for p in series], [f"month_{t:02d}" for t in times])
    scores = [float(a.highlight.max()) for a in analyses]
    flagged_area = [float(sum(r.area_px for r in a.regions)) for a in analyses]
    mu0, sd0 = float(np.mean(model.reference.normal_peaks)), float(np.std(model.reference.normal_peaks) + 1e-6)
    rep = longitudinal_report(times, scores, mu0, sd0, volumes=flagged_area)
    first = next((t for t, a in zip(times, analyses) if a.prediction != cfg.normal_class), None)
    rep["first_abnormal_prediction_at"] = first
    rep["per_timepoint"] = [{"month": t, "prediction": a.prediction, "p": a.probabilities[a.prediction],
                             "answered": a.answered, "true_lesion_radius_px": p.meta["lesion_radius"],
                             "flagged_area_px": ar}
                            for t, a, p, ar in zip(times, analyses, series, flagged_area)]
    for a, p in zip(analyses, series):
        (out / "reports" / f"mri_{a.case_id}.html").write_text(html_report(a, p.image, DISCLAIMER, truth=p.label))
    _write_json(out / "longitudinal.json", rep)
    _log(f"longitudinal: change detected at month {rep['change_detected_at']}, "
         f"trend {rep['trend_per_unit_time']:+.3f}/month (significant: {rep['trend_significant']})")
    return rep
