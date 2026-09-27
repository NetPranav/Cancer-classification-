"""Deterministic narrative: every sentence is generated from a number in the Analysis.

We deliberately do *not* let a free-running language model write the
explanation. Each clause below is filled in from a computed quantity (a
probability, a deciban weight, a region box, a stream step), so the
explanation cannot claim anything the analysis did not measure. An LLM may
later rephrase this text for readability (Roadmap Phase 9), but only with this
text as its sole source.
"""
from __future__ import annotations


def _pct(p: float) -> str:
    if p > 0.999:
        return ">99.9%"
    if 0 < p < 0.001:
        return "<0.1%"
    return f"{100 * p:.1f}%"


def narrate(a, cfg) -> str:
    L = []
    p_top = a.probabilities[a.prediction]
    L.append(f"FINDING: the image is most consistent with the pattern '{a.prediction}' "
             f"(calibrated probability {_pct(p_top)}).")
    if a.answered:
        L.append(f"DECISION: answered. Confidence {_pct(p_top)} clears the risk-controlled threshold "
                 f"{_pct(a.selective_threshold)}, which was set so that answered cases err at most "
                 f"{_pct(cfg.target_risk)} of the time (confidence {1 - cfg.delta:.0%}) on calibration data.")
    else:
        th = "no threshold could be certified on the available calibration data" if a.selective_threshold > 1 \
            else f"it is below the certified threshold {_pct(a.selective_threshold)}"
        L.append(f"DECISION: abstained, refer to a specialist. Confidence {_pct(p_top)} is not enough because {th}.")
    L.append(f"PLAUSIBLE SET ({1 - cfg.conformal_alpha:.0%} conformal): {', '.join(a.prediction_set)}.")

    L.append("")
    L.append("HOW THE IMAGE WAS READ (full-duplex stream):")
    stable = None
    for ev in a.stream:
        if ev["top"] == a.stream[-1]["top"] and stable is None:
            stable = ev["step"]
        elif ev["top"] != a.stream[-1]["top"]:
            stable = None
    L.append(f"  Read {a.tokens_read}/{a.tokens_total} patches, most suspicious first "
             f"({_pct(a.tokens_read / a.tokens_total)} of the image). The stream's belief settled on "
             f"'{a.stream[-1]['top']}' at step {stable}, and reading stopped when the halting head reached "
             f"{a.stream[-1]['halt']:.2f}" + (" (the calibrated stop threshold)." if a.tokens_read < a.tokens_total
                                                   else " (it never reached the stop threshold, so the whole image was read)."))
    for ev in a.stream[:3]:
        L.append(f"  step {ev['step']}: patch (row {ev['patch_row']}, col {ev['patch_col']}), saliency "
                 f"{ev['saliency']:.2f}x normal max -> belief '{ev['top']}' {_pct(ev['p_top'])}")

    L.append("")
    if a.regions:
        L.append(f"WHERE (candidate regions stronger than the median normal image's strongest deviation, "
                 f"{a.extra['region_threshold']:.2f}x; ranked by how much each one drives the conclusion):")
        base = a.component_probs["concept_evidence"][a.prediction]
        for r in a.regions[:5]:
            drop = base - (r.p_top_without if r.p_top_without is not None else base)
            role = ("DRIVES the conclusion" if drop > 0.2 else "contributes" if drop > 0.05
                    else "considered and dismissed (little effect on the conclusion)")
            L.append(f"  {r.region_id}: rows {r.bbox[0]}-{r.bbox[2]}, cols {r.bbox[1]}-{r.bbox[3]}, "
                     f"{r.area_px} px, peak {r.peak:.2f}x, dominated by '{r.dominant_concept}'; if it looked normal, "
                     f"concept evidence for '{a.prediction}' would go from {_pct(base)} to {_pct(r.p_top_without)}: {role}.")
    else:
        L.append("WHERE: no region deviates beyond the normal-reference threshold.")

    L.append("")
    L.append(f"WHY '{a.prediction}' RATHER THAN '{a.rival}' (weights of evidence, decibans; "
             f"prior {a.prior_db:+.1f} db):")
    groups = {"supporting": [], "opposing": [], "uninformative": []}
    for e in a.evidence:
        groups[e.verdict].append(e)
    labels = {"supporting": "Supporting", "opposing": "Counter-evidence considered and outweighed",
              "uninformative": "Considered and set aside (uninformative)"}
    for k in ("supporting", "opposing", "uninformative"):
        if groups[k]:
            L.append(f"  {labels[k]}:")
            for e in groups[k]:
                L.append(f"    {e.concept:<26} peak {e.value:5.2f}x normal max  {e.woe_db:+6.1f} db  ({e.description})")
    total = sum(e.woe_db for e in a.evidence) + a.prior_db
    L.append(f"  Net concept evidence: {total:+.1f} db (about {10 ** (total / 10):.3g}:1 odds from concepts alone).")

    L.append("")
    comp = ", ".join(f"{n} {_pct(p[a.prediction])}" for n, p in a.component_probs.items())
    L.append(f"CROSS-CHECKS (each channel on its own for '{a.prediction}'): {comp}. Fusion weights "
             + ", ".join(f"{k}={v:g}" for k, v in a.extra["weights"].items()) + ".")
    if a.nearest_support:
        L.append("  Most similar labelled reference cases: "
                 + "; ".join(f"#{s['support_index']} ({s['label']}, distance {s['distance']:.2f})"
                             for s in a.nearest_support) + ".")
    L.append("")
    L.append("LIMITS: pattern-level finding on this image only; it is not a diagnosis and not a validated "
             "prediction of future disease. Performance outside the calibration distribution "
             "(other scanners, stains, populations) is not guaranteed.")
    return "\n".join(L)
