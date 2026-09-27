# Training the General Pattern Model on Kaggle

Kaggle gives about 30 GPU-hours per week, with each session capped (check the
limit your account shows). Training therefore runs over several sessions, and
each session resumes from the last checkpoint.

## One-time setup
1. New notebook → **Settings**: Accelerator **GPU T4 x2**, **Internet on**.
2. **Add Input** → attach datasets. Good starting points (search by name and
   check the folder paths in the Input panel):
   - *Histopathologic Cancer Detection* (PCam, CSV labels, 96x96 tiles)
   - *Lung and Colon Cancer Histopathological Images* (LC25000, class folders)
   - *Brain Tumor MRI Dataset* (class folders, including `notumor`)
   - *Breast Ultrasound Images Dataset* (BUSI, masks beside images with `_mask`)
3. If this GitHub repository is **private**, the notebook cannot clone it.
   Either make it public, or download it as a zip and upload it as a Kaggle
   dataset, then attach it. The script detects an attached copy and
   installs from it instead of cloning.
4. Paste [`kaggle_train.py`](kaggle_train.py) into a cell. Edit `SOURCES`
   (uncomment and fix the paths), `PRESET` and `SESSION_LIMIT_H`.

## Each session
1. **Save Version → Save & Run All (Commit).** It runs in the background, so
   you can close the browser. The script clones the repo, restores the
   newest `gpm/checkpoint.pt` from any attached input, and trains until about
   45 minutes before the session limit. It checkpoints every 20 minutes
   (atomically, so a killed session never corrupts it). `/kaggle/working`
   becomes that version's output.
2. Next session: **Add Input → Your Work →** the previous version's output.
   The script finds its checkpoint automatically and continues: same
   optimiser state, same step count.

(Interactive runs work too, but they stop when the browser session idles.
Kaggle's exact limits change, so check your account's quota page.)

## Choosing the size
```
python -m oncopattern gpm-plan --gpu t4 --n-gpus 2 --hours 30
python -m oncopattern gpm-plan --gpu t4 --n-gpus 2 --hours 30 --unique-tokens 3.5e7   # with your data size
```
`base` (88M) fits the 30-hour budget. If memory runs out, halve `BATCH` and
double `GRAD_ACCUM`, or add `--grad-checkpoint`. Surprise conditioning costs
about 1.7x per step. `--no-surprise` turns it off, which is also the H6
ablation; `--stem linear` is the plain-ViT ablation.

## Source spec reference
| spec | layout |
|---|---|
| `phantom_tissue`, `phantom_mri` | synthetic, exact ground truth |
| `folder:<root>,modality=histology,mm=0.0005,rgb=1,normal=lung_n` | `root/<class>/*.png` (searched recursively) |
| `csv:<labels.csv>,images=<dir>,ext=.tif,id=id,label=label,names=0=normal\|1=metastasis` | CSV of ids and labels |
| `masks:<dir>,suffix=_mask,finding=tumour` or `masks:<images>,masks=<masks_dir>` | image and mask pairs |

Add `,weight=N` to any spec to sample it N times more often. `mm` is the
physical size of a pixel. Approximate values are fine; be consistent.

## After training
```
python -m oncopattern gpm-eval --model /kaggle/working/gpm/model.pt --n 30 --image-size 128
python -m oncopattern gpm-ask image.png "Is this tissue normal? Answer yes or no." \
    --model /kaggle/working/gpm/model.pt --modality histology --mm-per-px 0.0005 --context "H&E tissue."
```
