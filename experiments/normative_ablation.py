"""Does learning "normal" from healthy scans only make abnormalities easier to find? (Hypothesis H8)

Two identical tiny GPMs are trained on MRI phantoms for the same number of steps. They differ only in
which images the masked-patch ("what fits") objective sees:
  normative - healthy scans only        mixed - all scans, lesions included
Both then build a normative atlas from held-out healthy scans and are scored on unseen scans:
  image AUROC (lesion vs healthy), patch AUROC against true lesion masks, how many lesions get a
  highlighted patch, false highlights on healthy scans, and the causal effect of making the flagged
  regions healthy on P("yes, there is a lesion").

    python experiments/normative_ablation.py --steps 1500 --out outputs/normative_ablation.json
"""
import argparse
import json
import time

import numpy as np
import torch

from oncopattern.data.phantoms import mri_slice
from oncopattern.gpm.data import Example, ImageItem
from oncopattern.gpm.normative import NormativeAtlas
from oncopattern.gpm.tokenizer import ByteTokenizer
from oncopattern.gpm.train import load_model, train
from oncopattern.pipeline import auroc


def score(model_path, n_ref=60, n_test=40, seed=123):
    tok = ByteTokenizer()
    m = load_model(model_path)
    rng = np.random.default_rng(seed)
    healthy = [ImageItem(mri_slice(rng).image, 3.0, "phantom_mri") for _ in range(n_ref)]
    atlas = NormativeAtlas.fit(m, tok, healthy)
    img_s, img_y, pz, py, hits, false_flags, effects = [], [], [], [], 0, [], []
    p = m.cfg.patch_size
    for i in range(2 * n_test):
        lesion = i % 2 == 0
        ph = mri_slice(rng, lesion_radius=rng.uniform(1.5, 4.0) if lesion else 0.0)
        item = ImageItem(ph.image, 3.0, "phantom_mri")
        dev = atlas.deviation(m, tok, item)
        img_s.append(dev["score"])
        img_y.append(lesion)
        g = dev["z"].shape[0]
        truth = ph.mask[: g * p, : g * p].reshape(g, p, g, p).any((1, 3))
        pz.append(dev["z"].ravel())
        py.append(truth.ravel())
        if lesion:
            hits += bool((dev["flagged"] & truth).any())
            if i < 20:
                ex = Example([item], "Axial head MRI, 3 mm/px.", " Is there a mass lesion? Answer yes or no.")
                e = atlas.explain(m, tok, ex, ["yes", "no"])
                effects.append(e["p"][0] - e["p_if_all_flagged_healthy"][0])
        else:
            false_flags.append(dev["flagged"].sum())
    return {"image_auroc": auroc(img_s, img_y), "patch_auroc": auroc(np.concatenate(pz), np.concatenate(py)),
            "lesions_with_a_highlight_on_them": hits / n_test,
            "mean_false_highlighted_patches_per_healthy_scan": float(np.mean(false_flags)),
            "mean_drop_in_P_yes_when_flagged_regions_made_healthy": float(np.mean(effects)) if effects else None,
            "z_threshold": atlas.z_threshold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="outputs/normative_ablation.json")
    ap.add_argument("--workdir", default="outputs/normative_ablation")
    a = ap.parse_args()
    res = {}
    for name, normative in (("normative", True), ("mixed", False)):
        t = time.time()
        torch.manual_seed(0)
        train("tiny", steps=a.steps, batch_size=16, lr=1e-3, warmup=150, out=f"{a.workdir}/{name}",
              sources=["phantom_mri"], log_every=250, mim_fraction=0.4, normative=normative, resume=False)
        res[name] = score(f"{a.workdir}/{name}/model.pt")
        res[name]["train_minutes"] = (time.time() - t) / 60
        print(name, json.dumps(res[name], indent=1), flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
