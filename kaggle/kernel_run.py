"""Runs *inside* a Kaggle kernel. Pushed by ``kaggle/launch.py``, which prepends a ``CONFIG = {...}`` line.

1. Finds the code (attached private dataset ``<user>/oncopattern-code``) and puts it on the path.
2. Finds attached datasets by their folder layout (no hard-coded Kaggle paths) and builds source specs.
3. Restores the newest checkpoint from any attached input (the previous session's output).
4. Trains until the time budget, then evaluates on phantoms *and* the held-out split of every real dataset.
Everything is written to /kaggle/working/gpm, which becomes the kernel's output.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG") or {
    "preset": "tiny", "steps": 200, "batch_size": 8, "grad_accum": 1, "image_size": 64, "in_channels": 3,
    "time_budget_h": 0.2, "lr": 6e-4, "warmup": 100, "surprise": True, "eval_n": 10, "datasets": "auto",
}
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")  # overridable for local dry runs
INPUT, WORK = f"{ROOT}/input", f"{ROOT}/working"
OUT = f"{WORK}/gpm"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[kernel {time.time() - t0:7.0f}s] {msg}", flush=True)


# ---- 1. code ---------------------------------------------------------------------------------
code = sorted(glob.glob(f"{INPUT}/**/oncopattern/__init__.py", recursive=True))
if not code:  # the code folder may arrive as an archive, depending on how Kaggle stored the upload
    for arc in glob.glob(f"{INPUT}/**/oncopattern-code*.zip", recursive=True) + \
            glob.glob(f"{INPUT}/**/oncopattern-code*.tar", recursive=True):
        shutil.unpack_archive(arc, f"{WORK}/code")
    code = sorted(glob.glob(f"{WORK}/code/**/oncopattern/__init__.py", recursive=True))
if not code:
    sys.exit("oncopattern code not found in inputs: attach the <user>/oncopattern-code dataset")
CODE_DIR = os.path.dirname(os.path.dirname(code[0]))
ENV = dict(os.environ, PYTHONPATH=CODE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""),
           PYTHONDONTWRITEBYTECODE="1")
log(f"code: {CODE_DIR}")


# ---- 2. datasets, discovered by layout ------------------------------------------------------------
def find_dirs(name):
    return sorted(d for d in glob.glob(f"{INPUT}/**/{name}", recursive=True) if os.path.isdir(d))


def discover():
    specs = ["phantom_tissue,weight=1", "phantom_mri,weight=0.5"]
    for d in find_dirs("lung_image_sets")[:1]:  # LC25000 lung: lung_aca, lung_n, lung_scc
        specs.append(f"folder:{d},modality=histology,mm=0.0005,rgb=1,normal=lung_n,weight=2")
    for d in find_dirs("colon_image_sets")[:1]:  # LC25000 colon: colon_aca, colon_n
        specs.append(f"folder:{d},modality=histology,mm=0.0005,rgb=1,normal=colon_n,weight=1")
    for d in find_dirs("Training")[:1]:  # Brain Tumor MRI: glioma, meningioma, notumor, pituitary
        if os.path.isdir(os.path.join(d, "notumor")):
            specs.append(f"folder:{d},modality=mri,mm=0.5,weight=2")
    for d in find_dirs("Dataset_BUSI_with_GT")[:1]:  # BUSI: <class>/<name>.png + <name>_mask.png
        specs.append(f"masks:{d},suffix=_mask,modality=ultrasound,mm=0.1,finding=tumour,weight=1")
    for csv in glob.glob(f"{INPUT}/**/train_labels.csv", recursive=True)[:1]:  # PCam (competition data)
        tdir = os.path.join(os.path.dirname(csv), "train")
        if os.path.isdir(tdir):
            specs.append(f"csv:{csv},images={tdir},ext=.tif,names=0=normal|1=metastasis,"
                         f"modality=histology,mm=0.00097,rgb=1,weight=2")
    return specs


specs = CONFIG["datasets"] if CONFIG["datasets"] != "auto" else discover()
log("sources:\n  " + "\n  ".join(specs))

# ---- 3. resume ----------------------------------------------------------------------------------
if not os.path.exists(f"{OUT}/checkpoint.pt"):
    found = sorted(glob.glob(f"{INPUT}/**/gpm/checkpoint.pt", recursive=True), key=os.path.getmtime)
    if found:
        shutil.copy(found[-1], f"{OUT}/checkpoint.pt")
        for extra in ("train_log.jsonl",):
            src = os.path.join(os.path.dirname(found[-1]), extra)
            if os.path.exists(src):
                shutil.copy(src, f"{OUT}/{extra}")
        log(f"resuming from {found[-1]}")

# ---- 4. train + evaluate ----------------------------------------------------------------------------
try:
    import torch
    n_gpu = torch.cuda.device_count()
    gpus = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
except Exception:  # noqa: BLE001
    n_gpu, gpus = 0, []
log(f"GPUs: {gpus or 'none'}")
train_args = ["-m", "oncopattern", "gpm-train", "--preset", CONFIG["preset"], "--steps", str(CONFIG["steps"]),
              "--batch-size", str(CONFIG["batch_size"]), "--grad-accum", str(CONFIG["grad_accum"]),
              "--lr", str(CONFIG["lr"]), "--warmup", str(CONFIG["warmup"]),
              "--image-size", str(CONFIG["image_size"]), "--in-channels", str(CONFIG["in_channels"]),
              "--out", OUT, "--time-budget-h", str(CONFIG["time_budget_h"]), "--ckpt-minutes", "20",
              "--num-workers", str(CONFIG.get("num_workers", 2)), "--sources", *specs]
if not CONFIG.get("surprise", True):
    train_args.append("--no-surprise")
cmd = ([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(n_gpu)] if n_gpu > 1
       else [sys.executable]) + train_args
log(" ".join(cmd))
rc = subprocess.run(cmd, env=ENV, cwd=WORK).returncode
log(f"training exit code {rc}")

if os.path.exists(f"{OUT}/model.pt"):
    ev = subprocess.run([sys.executable, "-m", "oncopattern", "gpm-eval", "--model", f"{OUT}/model.pt",
                         "--n", str(CONFIG["eval_n"]), "--image-size", str(CONFIG["image_size"]),
                         "--out", f"{OUT}/eval.json", "--sources", *specs], env=ENV, cwd=WORK)
    log(f"evaluation exit code {ev.returncode}")

with open(f"{OUT}/run_info.json", "w") as fh:
    json.dump({"config": CONFIG, "sources": specs, "gpus": gpus, "train_exit_code": rc,
               "wall_seconds": time.time() - t0}, fh, indent=2)
log("done")
