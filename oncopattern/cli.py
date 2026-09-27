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

    g = sub.add_parser("gpm-plan", help="what General Pattern Model size fits a GPU budget and dataset")
    g.add_argument("--gpu", default="t4")
    g.add_argument("--n-gpus", type=int, default=2)
    g.add_argument("--hours", type=float, default=30)
    g.add_argument("--mfu", type=float, default=0.3, help="assumed hardware utilisation")
    g.add_argument("--unique-tokens", type=float, default=None, help="unique training tokens in your data")

    t = sub.add_parser("gpm-train", help="train the General Pattern Model from scratch (resumable)")
    t.add_argument("--preset", default="tiny")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=150)
    t.add_argument("--out", default="outputs/gpm")
    t.add_argument("--sources", nargs="+", default=["phantom_tissue", "phantom_mri"],
                   help="phantom_tissue | phantom_mri | folder:<root>,modality=,mm=,rgb=1 | "
                        "csv:<labels.csv>,images=,ext=,names=0=normal|1=tumour | masks:<imgs>,masks=<dir>|suffix=_mask "
                        "(see oncopattern/gpm/train.py:build_sources)")
    t.add_argument("--image-size", type=int, default=64)
    t.add_argument("--in-channels", type=int, default=1, help="3 for colour (H&E) data")
    t.add_argument("--time-budget-h", type=float, default=None, help="stop and checkpoint before this many hours")
    t.add_argument("--ckpt-minutes", type=float, default=20)
    t.add_argument("--num-workers", type=int, default=0)
    t.add_argument("--grad-checkpoint", action="store_true")
    t.add_argument("--no-surprise", action="store_true", help="disable surprise conditioning (ablation; ~1.7x cheaper)")
    t.add_argument("--stem", default=None, choices=["d4conv", "d4", "linear"], help="patch stem (ablation)")
    t.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("gpm-eval", help="grade every task with the verifier; surprise-map AUROC")
    e.add_argument("--model", required=True)
    e.add_argument("--n", type=int, default=20, help="examples per task")
    e.add_argument("--out", default=None, help="write results JSON here")
    e.add_argument("--image-size", type=int, default=64)

    q = sub.add_parser("gpm-ask", help="ask the General Pattern Model anything about an image")
    q.add_argument("image")
    q.add_argument("question")
    q.add_argument("--model", required=True)
    q.add_argument("--modality", default="other")
    q.add_argument("--mm-per-px", type=float, default=1.0)
    q.add_argument("--context", default="", help="text before the image, e.g. 'H&E tissue, 0.5 um/px.'")
    q.add_argument("--image-size", type=int, default=64)

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
    elif args.cmd == "gpm-plan":
        from oncopattern.gpm.compute import format_plan, plan
        print(format_plan(plan(args.gpu, args.n_gpus, args.hours, args.mfu, args.unique_tokens)))
    elif args.cmd == "gpm-train":
        from oncopattern.gpm.train import train
        res = train(args.preset, args.steps, args.batch_size, args.lr, args.warmup, args.out, args.sources,
                    args.time_budget_h, args.ckpt_minutes, seed=args.seed, grad_accum=args.grad_accum,
                    image_size=args.image_size, num_workers=args.num_workers,
                    overrides={"grad_checkpoint": args.grad_checkpoint, "in_channels": args.in_channels,
                               "surprise_conditioning": not args.no_surprise,
                               **({"stem": args.stem} if args.stem else {})})
        print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))
    elif args.cmd == "gpm-eval":
        from oncopattern.gpm.evaluate import evaluate_tasks, surprise_auroc
        from oncopattern.gpm.sources import MRIPhantomSource, TissuePhantomSource
        from oncopattern.gpm.tokenizer import ByteTokenizer
        from oncopattern.gpm.train import load_model
        tok = ByteTokenizer()
        model = load_model(args.model)
        srcs = [TissuePhantomSource(tok, size=args.image_size), MRIPhantomSource(tok, size=args.image_size)]
        res = {"tasks": evaluate_tasks(model, tok, srcs, args.n),
               "surprise": surprise_auroc(model, tok, size=args.image_size)}
        text = json.dumps(res, indent=2, default=float)
        if args.out:
            Path(args.out).write_text(text)
        print(text)
    elif args.cmd == "gpm-ask":
        from oncopattern.data.io import load, prepare
        from oncopattern.gpm.data import Example, ImageItem, prompt_batch
        from oncopattern.gpm.tokenizer import ByteTokenizer
        from oncopattern.gpm.train import load_model
        tok = ByteTokenizer()
        model = load_model(args.model)
        img = prepare(load(args.image), args.image_size)
        ex = Example([ImageItem(img, args.mm_per_px, args.modality)], args.context, " " + args.question)
        print(tok.decode(model.generate(prompt_batch(ex, tok, model.cfg.patch_size), 32, eos_id=tok.eos_id)))
    elif args.cmd == "catalogue":
        from oncopattern.data.catalogue import search
        for e in search(organ=args.organ, modality=args.modality, longitudinal=True if args.longitudinal else None):
            print(f"{e.name:<36} {e.domain:<11} {','.join(e.modalities):<16} {e.subjects:<22} "
                  f"{'longitudinal' if e.longitudinal else '':<12} {e.access}")


if __name__ == "__main__":
    main()
