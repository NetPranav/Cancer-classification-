"""Command line: ``python -m oncopattern <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oncopattern", description="Pattern-level cancer image analysis (research).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run every phase end to end on phantoms with ground truth")
    d.add_argument("--out", default="outputs/demo")
    d.add_argument("--quick", action="store_true", help="smaller data, ~20 s")
    d.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("analyze", help="analyse one image with a saved model and write an HTML report")
    a.add_argument("image")
    a.add_argument("--model", required=True)
    a.add_argument("--out", default="report.html")
    a.add_argument("--invert", action="store_true", help="invert intensities (make nuclei dark)")

    c = sub.add_parser("catalogue", help="search the public dataset catalogue")
    c.add_argument("--organ")
    c.add_argument("--modality")
    c.add_argument("--longitudinal", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "demo":
        from oncopattern.demo import run_longitudinal_demo, run_tissue_demo
        res = run_tissue_demo(Path(args.out) / "tissue", args.quick, args.seed)
        lon = run_longitudinal_demo(Path(args.out) / "mri_longitudinal", args.quick, args.seed + 1)
        m = res["metrics"]
        print(json.dumps({k: m[k] for k in ("accuracy_all", "coverage_answered", "accuracy_on_answered",
                                            "conformal_coverage", "auroc_abnormal_vs_normal", "pixel_auroc",
                                            "pointing_game", "mean_fraction_of_image_read")}, indent=2))
        print("failure recommendations:", json.dumps(res["failures"]["recommendations"], indent=2))
        print("longitudinal change detected at month:", lon["change_detected_at"])
    elif args.cmd == "analyze":
        from oncopattern.data.io import load, prepare
        from oncopattern.pipeline import DISCLAIMER, OncoPattern
        from oncopattern.reasoning.report import html_report
        model = OncoPattern.load(args.model)
        img = load(args.image)
        if img.ndim != 2:
            sys.exit("analyze expects a 2D image; use oncopattern.data.io.axial_slices for volumes")
        img = prepare(img, model.cfg.image_size, args.invert)
        res = model.analyze(img, Path(args.image).stem)
        Path(args.out).write_text(html_report(res, img, DISCLAIMER))
        print(res.narrative)
        print(f"\nreport: {args.out}")
    elif args.cmd == "catalogue":
        from oncopattern.data.catalogue import search
        for e in search(organ=args.organ, modality=args.modality, longitudinal=True if args.longitudinal else None):
            print(f"{e.name:<36} {e.domain:<11} {','.join(e.modalities):<16} {e.subjects:<22} "
                  f"{'longitudinal' if e.longitudinal else '':<12} {e.access}")


if __name__ == "__main__":
    main()
