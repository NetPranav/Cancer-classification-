"""Paste this into a Kaggle notebook cell (Accelerator: GPU T4 x2, Internet: on). See kaggle/README.md.

It installs the repository, restores the previous session's checkpoint if you
attached that notebook's output as an input, trains until the session budget
is nearly used, and leaves everything in /kaggle/working. Save the notebook
version to keep it.
"""
import glob
import os
import shutil
import subprocess
import sys

# ---- edit these -------------------------------------------------------------
REPO = "https://github.com/NetPranav/Cancer-classification-.git"
BRANCH = "claude/cancer-pattern-detection-roadmap-78nnf0"
PRESET = "base"  # run `python -m oncopattern gpm-plan` to choose
STEPS = 60000  # total across all sessions; training resumes where it stopped
BATCH = 32  # per GPU; lower it (and raise GRAD_ACCUM) if you run out of memory
GRAD_ACCUM = 1
IMAGE_SIZE = 128  # the base preset uses 16 px patches -> 64 patches per image
IN_CHANNELS = 3  # colour, for H&E; grey sources are repeated to 3 channels automatically
SESSION_LIMIT_H = 12.0  # check the limit Kaggle shows for your account
OUT = "/kaggle/working/gpm"
SOURCES = [
    # synthetic data with exact answers: keeps the verifier tasks and grounding honest
    "phantom_tissue,weight=1",
    # examples; uncomment the datasets you attached (paths are under /kaggle/input/...)
    # "folder:/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets,"
    #     "modality=histology,mm=0.0005,rgb=1,normal=lung_n,weight=3",
    # "csv:/kaggle/input/histopathologic-cancer-detection/train_labels.csv,"
    #     "images=/kaggle/input/histopathologic-cancer-detection/train,ext=.tif,"
    #     "names=0=normal|1=metastasis,modality=histology,mm=0.00097,rgb=1,weight=3",
    # "folder:/kaggle/input/brain-tumor-mri-dataset/Training,modality=mri,mm=0.5,weight=2",
    # "masks:/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT,suffix=_mask,"
    #     "modality=ultrasound,mm=0.1,finding=tumour,weight=1",
]
# ------------------------------------------------------------------------------

if not os.path.exists("/kaggle/working/repo"):
    # a private repo cannot be cloned: upload it as a Kaggle dataset instead and it is picked up here
    uploaded = [os.path.dirname(p) for p in glob.glob("/kaggle/input/*/**/oncopattern/__init__.py", recursive=True)]
    if uploaded:
        shutil.copytree(os.path.dirname(uploaded[0]), "/kaggle/working/repo")
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, "/kaggle/working/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/kaggle/working/repo"], check=True)

# resume: copy the newest checkpoint from any attached input (a previous version's output)
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(f"{OUT}/checkpoint.pt"):
    found = sorted(glob.glob("/kaggle/input/**/gpm/checkpoint.pt", recursive=True), key=os.path.getmtime)
    if found:
        shutil.copy(found[-1], f"{OUT}/checkpoint.pt")
        print("resuming from", found[-1])

n_gpu = int(subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.device_count())"],
                           capture_output=True, text=True).stdout.strip() or 0)
budget = SESSION_LIMIT_H - 0.75  # leave time to save the notebook version
args = ["-m", "oncopattern", "gpm-train", "--preset", PRESET, "--steps", str(STEPS), "--batch-size", str(BATCH),
        "--grad-accum", str(GRAD_ACCUM), "--image-size", str(IMAGE_SIZE), "--in-channels", str(IN_CHANNELS),
        "--out", OUT, "--time-budget-h", str(budget), "--ckpt-minutes", "20", "--num-workers", "3",
        "--sources", *SOURCES]
if n_gpu > 1:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(n_gpu)] + args
else:
    cmd = [sys.executable] + args
print(" ".join(cmd))
subprocess.run(cmd, check=True, cwd="/kaggle/working/repo")
subprocess.run([sys.executable, "-m", "oncopattern", "gpm-eval", "--model", f"{OUT}/model.pt", "--n", "30",
                "--image-size", str(IMAGE_SIZE), "--out", f"{OUT}/eval.json"], cwd="/kaggle/working/repo")
