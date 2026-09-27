"""Drive Kaggle GPU runs from the command line (no notebook clicking).

Credentials (never committed, never printed): the environment variables
``KAGGLE_USERNAME`` plus either ``KAGGLE_API_TOKEN`` (new-style token) or
``KAGGLE_KEY`` (legacy kaggle.json key). A git-ignored ``.env`` file at the
repository root is also read.

    python kaggle/launch.py check                  # auth, GPU quota, dataset slugs
    python kaggle/launch.py upload-code            # committed code -> private dataset <user>/oncopattern-code
    python kaggle/launch.py push --mode smoke      # ~15 min on 2x T4: memory, throughput, does loss fall?
    python kaggle/launch.py status
    python kaggle/launch.py fetch                  # download outputs to outputs/kaggle/<time>/ and summarise
    python kaggle/launch.py push --mode full --steps <from fetch>   # one session; re-run to continue
    python kaggle/launch.py push --dry-run ...     # build the kernel folder only (no network)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "kaggle" / "_build"
KERNEL_SLUG = "oncopattern-gpm"
CODE_SLUG = "oncopattern-code"
DATASETS = [  # public Kaggle datasets; the kernel finds them by folder layout, not by path
    "andrewmvd/lung-and-colon-cancer-histopathological-images",  # LC25000
    "masoudnickparvar/brain-tumor-mri-dataset",
    "aryashah2k/breast-ultrasound-images-dataset",  # BUSI (with masks)
]
PCAM_COMPETITION = "histopathologic-cancer-detection"  # requires accepting the rules on kaggle.com first

MODES = {
    # short run to measure memory and seconds/step before spending hours
    "smoke": dict(preset="base", steps=100000, batch_size=16, grad_accum=1, image_size=128, in_channels=3,
                  time_budget_h=0.25, lr=6e-4, warmup=200, surprise=True, eval_n=10, num_workers=2),
    "full": dict(preset="base", steps=0, batch_size=16, grad_accum=1, image_size=128, in_channels=3,
                 time_budget_h=11.0, lr=6e-4, warmup=1000, surprise=True, eval_n=40, num_workers=2),
}


def load_env():
    f = REPO / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("KAGGLE_USERNAME"):
        sys.exit("KAGGLE_USERNAME is not set (environment or .env)")
    if not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")):
        sys.exit("set KAGGLE_API_TOKEN (or legacy KAGGLE_KEY) in the environment or .env")


def user() -> str:
    return os.environ["KAGGLE_USERNAME"]


def kaggle(*args, check=True) -> str:
    exe = shutil.which("kaggle") or sys.exit("kaggle CLI missing: pip install kaggle")
    res = subprocess.run([exe, *args], capture_output=True, text=True, env=os.environ)
    out = (res.stdout + res.stderr).strip()
    for secret in (os.environ.get("KAGGLE_API_TOKEN"), os.environ.get("KAGGLE_KEY")):
        if secret:
            out = out.replace(secret, "***")
    if check and res.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


# ------------------------------------------------------------------------------------------------ commands
def cmd_check(a):
    load_env()
    print(kaggle("config", "view", check=False))
    print(kaggle("quota", check=False))
    for d in DATASETS:
        out = kaggle("datasets", "files", d, check=False)
        print(f"{d}: {'OK' if 'Traceback' not in out and '404' not in out else 'NOT FOUND'}")
    out = kaggle("competitions", "files", PCAM_COMPETITION, check=False)
    print(f"{PCAM_COMPETITION}: {'OK' if '403' not in out and 'Traceback' not in out else 'rules not accepted?'}")


def cmd_upload_code(a):
    load_env()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    if dirty.strip():
        print("warning: uncommitted changes are NOT uploaded (git archive HEAD)")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    d = BUILD / "code"
    shutil.rmtree(d, ignore_errors=True)
    (d / "oncopattern-code").mkdir(parents=True)
    tar = BUILD / "code.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar), "HEAD", "oncopattern", "pyproject.toml",
                    "README.md"], cwd=REPO, check=True)
    with tarfile.open(tar) as t:
        t.extractall(d / "oncopattern-code", filter="data")
    tar.unlink()
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": CODE_SLUG, "id": f"{user()}/{CODE_SLUG}", "licenses": [{"name": "other"}]}, indent=2))
    exists = "error" not in kaggle("datasets", "status", f"{user()}/{CODE_SLUG}", check=False).lower()
    if exists:
        print(kaggle("datasets", "version", "-p", str(d), "-m", f"code {sha}", "-r", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(d), "-r", "zip"))  # private by default
    print(f"code {sha} -> {user()}/{CODE_SLUG} (private)")


def build_kernel(mode: str, overrides: dict, pcam: bool, resume: bool, username: str) -> Path:
    cfg = {**MODES[mode], **{k: v for k, v in overrides.items() if v is not None}, "datasets": "auto"}
    if mode == "full" and not cfg["steps"]:
        sys.exit("full mode needs --steps (run smoke, then `fetch` prints the right number)")
    d = BUILD / "kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    body = (REPO / "kaggle" / "kernel_run.py").read_text()
    (d / "kernel_run.py").write_text(f"CONFIG = {json.dumps(cfg)}\n" + body)
    meta = {"id": f"{username}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "kernel_run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{username}/{CODE_SLUG}", *DATASETS],
            "competition_sources": [PCAM_COMPETITION] if pcam else [],
            # the previous version's output (checkpoint) is attached as an input to resume
            "kernel_sources": [f"{username}/{KERNEL_SLUG}"] if resume else [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return d


def cmd_push(a):
    if not a.dry_run:
        load_env()
    d = build_kernel(a.mode, {"preset": a.preset, "steps": a.steps, "time_budget_h": a.hours,
                              "batch_size": a.batch_size, "image_size": a.image_size},
                     a.pcam, a.resume, os.environ.get("KAGGLE_USERNAME", "USER"))
    print(f"kernel folder: {d}")
    if a.dry_run:
        print((d / "kernel-metadata.json").read_text())
        return
    timeout = int((json.loads((d / "kernel_run.py").read_text().split("\n", 1)[0][9:])["time_budget_h"] + 0.75) * 3600)
    print(kaggle("kernels", "push", "-p", str(d), "--timeout", str(timeout)))
    print(f"pushed; watch with: python kaggle/launch.py status")


def cmd_status(a):
    load_env()
    print(kaggle("kernels", "status", f"{user()}/{KERNEL_SLUG}", check=False))


def summarise(out: Path):
    info = {}
    log = next(out.rglob("train_log.jsonl"), None)
    if log:
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        sps = sorted(r["sec_per_step"] for r in recs[-20:])
        info["steps_done"] = recs[-1]["step"]
        info["last_loss"] = recs[-1]["loss"]
        info["sec_per_step_median"] = sps[len(sps) // 2]
        budget_h = 26.0  # of a 30 h weekly quota, leaving room for evaluation and a smoke run
        info["suggested_full_steps_for_26h"] = int(budget_h * 3600 / info["sec_per_step_median"])
    ev = next(out.rglob("eval.json"), None)
    if ev:
        e = json.loads(ev.read_text())
        info["eval"] = {k: {"reward": round(v["mean_reward"], 3), "baseline": round(v["baseline_reward"], 3)}
                        for k, v in e.get("tasks", {}).items()}
        info["surprise"] = e.get("surprise")
    ri = next(out.rglob("run_info.json"), None)
    if ri:
        info["run_info"] = json.loads(ri.read_text())
    print(json.dumps(info, indent=2))


def cmd_fetch(a):
    load_env()
    out = REPO / "outputs" / "kaggle" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    print(kaggle("kernels", "output", f"{user()}/{KERNEL_SLUG}", "-p", str(out), check=False)[-2000:])
    summarise(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("upload-code")
    p = sub.add_parser("push")
    p.add_argument("--mode", choices=list(MODES), default="smoke")
    p.add_argument("--preset")
    p.add_argument("--steps", type=int)
    p.add_argument("--hours", type=float, help="training time budget for this session")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--pcam", action="store_true", help="attach PCam (accept the competition rules first)")
    p.add_argument("--resume", action="store_true", help="attach the previous version's output to continue")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    sub.add_parser("fetch")
    s = sub.add_parser("summarise")
    s.add_argument("dir")
    a = ap.parse_args(argv)
    {"check": cmd_check, "upload-code": cmd_upload_code, "push": cmd_push, "status": cmd_status,
     "fetch": cmd_fetch, "summarise": lambda a: summarise(Path(a.dir))}[a.cmd](a)


if __name__ == "__main__":
    main()
