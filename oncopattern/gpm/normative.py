"""Normative atlas: learn what *healthy* looks like, then explain a scan as its deviation from health.

This is normative modelling (as used for brain-growth charts), done by the
General Pattern Model itself.

1. **Healthy grammar.** With ``Mixture(normative=True)`` the masked-patch
   objective sees only healthy images, so "what fits here" means "what healthy
   anatomy looks like here".

2. **Atlas calibration.** Some places are hard to predict even in healthy
   people (tissue edges, skull, ventricles). Raw prediction error is therefore
   not a fair abnormality score. Over held-out healthy scans we estimate, for
   every patch position p, the mean and spread of log prediction error, shrunk
   toward the global value when data are few:

       mu_p = (n mean_p + k mu) / (n + k),   var_p = (n var_p + k var) / (n + k)
       z_p(x) = (log err_p(x) - mu_p) / sd_p          ("sigmas from healthy")

   A scan's score is the mean of its top-3 z. Its **percentile among healthy
   references** says how unusual it is ("more unusual than 99.6% of healthy
   brains in the reference set").

3. **Healthy counterfactual.** For flagged patches the model redraws them as
   it expects healthy tissue to look *given the surrounding patient-specific
   context*: hide them and use the masked-patch head. Original minus redrawn
   shows *what* is abnormal, not just where.

4. **Why the answer.** The question is asked again on the counterfactual
   image, for all flagged regions together and for each region alone:

       effect(R) = P(answer | scan) - P(answer | scan with R made healthy)

   A region with a large effect is the reason for the answer. This is a causal
   test of the model's own decision, not a saliency heuristic.

Positions are grid positions, so references and patients should share a
roughly common frame (same field of view, centred anatomy). Proper
registration to an anatomical template is on the roadmap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage as ndi

from oncopattern.calibration.conformal import softmax
from oncopattern.gpm.data import Example, ImageItem, answer_batch, build, collate
from oncopattern.gpm.model import GeneralPatternModel
from oncopattern.gpm.tokenizer import ByteTokenizer


def _mim_example(item: ImageItem) -> Example:
    return Example([item], "Study this image.", mim=True, task="mim", meta={"task": "mim"})


@torch.no_grad()
def patch_log_errors(model: GeneralPatternModel, tok: ByteTokenizer, items: list[ImageItem], groups: int = 4,
                     seed: int = 0, device=None, batch_size: int = 16) -> list[np.ndarray]:
    """log prediction error per patch (each patch hidden once), for each image."""
    model.eval()
    out = []
    for i in range(0, len(items), batch_size):
        built = [build(_mim_example(it), tok, model.cfg.patch_size) for it in items[i:i + batch_size]]
        b = collate(built, tok, device)
        err = model._prediction_error(b, groups, torch.Generator().manual_seed(seed)).cpu().numpy()
        off = 0
        for bb in built:
            n = len(bb["patches"])
            out.append(np.log(err[off:off + n] + 1e-6))
            off += n
    return out


@dataclass
class NormativeAtlas:
    grid: tuple = (0, 0)
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    ref_scores: np.ndarray | None = None  # image scores of held-out healthy references
    z_threshold: float = 3.0
    z_grow: float = 1.3  # hysteresis: regions start above z_threshold and grow through patches above z_grow
    n_reference: int = 0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, model, tok, healthy: list[ImageItem], shrink: float = 8.0, holdout: float = 0.3,
            device=None, **meta) -> "NormativeAtlas":
        """Fit on healthy scans; a held-out part sets the image-score distribution and highlight threshold."""
        errs = np.stack(patch_log_errors(model, tok, healthy, device=device))  # (N, P)
        n_hold = max(2, int(len(errs) * holdout))
        fit_e, hold_e = errs[n_hold:], errs[:n_hold]
        n = len(fit_e)
        mu_all, var_all = fit_e.mean(), fit_e.var()
        mu_p, var_p = fit_e.mean(0), fit_e.var(0)
        mu = (n * mu_p + shrink * mu_all) / (n + shrink)
        var = (n * var_p + shrink * var_all) / (n + shrink)
        p = model.cfg.patch_size
        h, w = healthy[0].pixels.shape[-2:]
        atlas = cls(grid=(h // p, w // p), mu=mu, sd=np.sqrt(var + 1e-6), n_reference=len(healthy), meta=meta)
        z_hold = (hold_e - atlas.mu) / atlas.sd
        atlas.z_threshold = float(max(np.quantile(z_hold, 0.995), 2.0))  # aim: ~1 false patch in 200 on healthy scans
        atlas.meta["false_patch_rate"] = float((z_hold > atlas.z_threshold).mean())
        atlas.z_grow = float(min(max(np.quantile(z_hold, 0.90), 0.5), atlas.z_threshold))
        atlas.ref_scores = np.sort([cls._score(z) for z in z_hold])
        return atlas

    @staticmethod
    def _score(z: np.ndarray) -> float:
        return float(np.sort(z)[-3:].mean())

    # ------------------------------------------------------------------ use
    def deviation(self, model, tok, item: ImageItem, device=None) -> dict:
        z = (patch_log_errors(model, tok, [item], device=device)[0] - self.mu) / self.sd
        score = self._score(z)
        pct = float(np.searchsorted(self.ref_scores, score) / len(self.ref_scores))
        zmap = z.reshape(self.grid)
        # hysteresis (as in Canny edge detection): a region must contain a strongly deviating patch, and
        # then extends through connected moderately deviating ones, so the whole lesion is covered
        # while false alarms on healthy scans still need a strong seed
        seeds = zmap > self.z_threshold
        flagged = ndi.binary_propagation(seeds, structure=np.ones((3, 3)), mask=zmap > self.z_grow) | seeds
        lab, n = ndi.label(flagged, structure=np.ones((3, 3)))
        regions = []
        p = model.cfg.patch_size
        for r in range(1, n + 1):
            m = lab == r
            ys, xs = np.nonzero(m)
            regions.append({"id": f"R{r}", "patches": int(m.sum()), "peak_z": float(zmap[m].max()),
                            "box_px": [int(ys.min() * p), int(xs.min() * p), int((ys.max() + 1) * p), int((xs.max() + 1) * p)],
                            "mask": m})
        regions.sort(key=lambda r: -r["peak_z"])
        for i, r in enumerate(regions, 1):
            r["id"] = f"R{i}"
        return {"z": zmap, "score": score, "percentile_vs_healthy": pct, "flagged": flagged, "regions": regions}

    @torch.no_grad()
    def healthy_counterfactual(self, model, tok, item: ImageItem, patch_mask: np.ndarray, device=None) -> np.ndarray:
        """Redraw the patches in ``patch_mask`` (grid bool) as the model expects healthy tissue to look."""
        pix = item.pixels.astype(np.float32).copy()
        if not patch_mask.any():
            return pix
        b = collate([build(_mim_example(item), tok, model.cfg.patch_size)], tok, device)
        m = torch.from_numpy(patch_mask.ravel()).to(b["patches"].device)
        x = model.embed(b, m, model.sur_mean.expand(len(m)) if model.cfg.surprise_conditioning else None)
        h = model.trunk(x, model.attention_mask(b["block"], b["valid"]))
        pidx = b["patch_index"]
        hp = torch.empty(len(m), h.shape[-1], dtype=h.dtype, device=h.device)
        hp[pidx[pidx >= 0]] = h[pidx >= 0]
        c = model.cfg.in_channels
        p = model.cfg.patch_size
        pred = model.patch_head(hp[m]).float().clamp(0, 1).view(-1, c, p, p).cpu().numpy()
        img = pix if pix.ndim == 3 else pix[None]
        if img.shape[0] != c:  # grey image, colour model: the model draws colour; show its mean
            pred = pred.mean(1, keepdims=True)
        for k, (gy, gx) in enumerate(zip(*np.nonzero(patch_mask))):
            img[:, gy * p:(gy + 1) * p, gx * p:(gx + 1) * p] = pred[k]
        return img if pix.ndim == 3 else img[0]

    def explain(self, model, tok, ex: Example, options: list[str], device=None) -> dict:
        """Deviation map + healthy counterfactual + causal effect of each region on the answer."""
        item = ex.images[0]
        dev = self.deviation(model, tok, item, device)

        def probs(it):
            e = Example([it, *ex.images[1:]], ex.prompt, ex.question, "", ex.task, {"options": options})
            with torch.no_grad():
                return softmax(model.sequence_logprob(answer_batch(e, options, tok, model.cfg.patch_size, device))
                               .cpu().numpy())

        p0 = probs(item)
        top = int(p0.argmax())
        healthy_all = self.healthy_counterfactual(model, tok, item, dev["flagged"], device)
        p_all = probs(ImageItem(healthy_all, item.spacing_mm, item.modality, item.t_days))
        for r in dev["regions"][:6]:
            cf = self.healthy_counterfactual(model, tok, item, r["mask"], device)
            pr = probs(ImageItem(cf, item.spacing_mm, item.modality, item.t_days))
            r["p_answer_if_healthy"] = float(pr[top])
            r["effect"] = float(p0[top] - pr[top])
        return {**dev, "options": options, "answer": options[top], "p": p0.tolist(), "threshold": self.z_threshold,
                "p_if_all_flagged_healthy": p_all.tolist(), "healthy_image": healthy_all,
                "narrative": self.narrate(dev, options, p0, p_all)}

    def narrate(self, dev, options, p0, p_all) -> str:
        top = int(np.argmax(p0))
        lines = [f"Answer: '{options[top]}' (probability {100 * p0[top]:.1f}%).",
                 f"Compared with {self.n_reference} healthy reference scans, this scan is more unusual than "
                 f"{100 * dev['percentile_vs_healthy']:.1f}% of them (deviation score {dev['score']:.1f} sigma).",
                 f"Highlighted: regions containing a patch more than {self.z_threshold:.1f} sigma from healthy "
                 f"(on held-out healthy scans such patches are {100 * self.meta.get('false_patch_rate', 0):.2f}%), "
                 f"extended through connected patches above {self.z_grow:.1f} sigma."]
        if not dev["regions"]:
            lines.append("No single region crosses the threshold" + (
                "; the scan is unusual overall rather than in one place." if dev["percentile_vs_healthy"] > 0.95
                else "; the scan looks like the healthy references."))
        for r in dev["regions"][:6]:
            y0, x0, y1, x1 = r["box_px"]
            eff = r.get("effect")
            why = "" if eff is None else (
                f" If this region looked healthy, P('{options[top]}') would go from {100 * p0[top]:.1f}% to "
                f"{100 * r['p_answer_if_healthy']:.1f}%" + (": this region is a reason for the answer." if eff > 0.2
                                                           else ": it matters little for the answer."))
            lines.append(f"{r['id']}: rows {y0}-{y1}, cols {x0}-{x1}, peak {r['peak_z']:.1f} sigma from healthy.{why}")
        if dev["regions"]:
            lines.append(f"With every highlighted region made healthy, P('{options[top]}') would be "
                         f"{100 * p_all[top]:.1f}%.")
        lines.append("Research software: not a diagnosis; the healthy reference set defines what 'normal' means.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        torch.save({k: v for k, v in self.__dict__.items()}, path)

    @classmethod
    def load(cls, path: str) -> "NormativeAtlas":
        return cls(**torch.load(path, weights_only=False))


def explanation_html(expl: dict, image: np.ndarray, title: str = "Normative explanation") -> str:
    """Original | deviation-from-healthy overlay | model's healthy version | difference, plus the reasoning."""
    import html

    from oncopattern.reasoning.report import CSS, _heat, png_bytes, _uri

    grey = image.mean(0) if image.ndim == 3 else image
    healthy = expl["healthy_image"]
    healthy = healthy.mean(0) if healthy.ndim == 3 else healthy
    s = 5
    up = lambda a: np.kron(a, np.ones((s, s, 1)))
    g = np.repeat(np.clip(grey, 0, 1)[..., None] * 255, 3, -1)
    zmap = np.kron(np.clip(expl["z"], 0, None), np.ones((grey.shape[0] // expl["z"].shape[0],) * 2))
    zmap = zmap[: grey.shape[0], : grey.shape[1]]
    thr = max(expl.get("threshold", 3.0), 1e-6)
    heat = _heat(zmap / (2 * thr))
    alpha = np.clip(zmap / thr - 0.5, 0, 1)[..., None] * 0.7
    over = g * (1 - alpha) + heat * alpha
    for r in expl["regions"]:
        y0, x0, y1, x1 = r["box_px"]
        over[y0, x0:x1] = over[y1 - 1, x0:x1] = [0, 200, 255]
        over[y0:y1, x0] = over[y0:y1, x1 - 1] = [0, 200, 255]
    hl = np.repeat(np.clip(healthy, 0, 1)[..., None] * 255, 3, -1)
    diff = _heat(np.abs(grey - healthy) / 0.5)
    imgs = [("Scan", g), ("Deviation from healthy (boxes: flagged regions)", over),
            ("Model's healthy version of this scan", hl), ("Difference: what is abnormal", diff)]
    figs = "".join(f"<figure><img src='{_uri(png_bytes(up(a)))}' alt='{html.escape(c)}'><figcaption>{html.escape(c)}"
                   f"</figcaption></figure>" for c, a in imgs)
    rows = "".join(f"<tr><td>{r['id']}</td><td>{r['box_px']}</td><td class='num'>{r['peak_z']:.1f}</td>"
                   f"<td class='num'>{100 * r.get('p_answer_if_healthy', float('nan')):.1f}%</td>"
                   f"<td class='num'>{100 * r.get('effect', float('nan')):+.1f} pts</td></tr>" for r in expl["regions"][:6])
    probs = "".join(f"<tr><td>{html.escape(o)}</td><td class='num'>{100 * p:.1f}%</td>"
                    f"<td class='num'>{100 * q:.1f}%</td></tr>"
                    for o, p, q in zip(expl["options"], expl["p"], expl["p_if_all_flagged_healthy"]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}
.imgs img{{width:min(230px,100%)}}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<div class="muted">more unusual than {100 * expl['percentile_vs_healthy']:.1f}% of healthy reference scans</div>
<h2>What the model compared</h2><div class="imgs card">{figs}</div>
<h2>Answer, and the answer if the flagged regions were healthy</h2>
<div class="card table-wrap"><table><tr><th>option</th><th class="num">as scanned</th><th class="num">flagged regions made healthy</th></tr>{probs}</table></div>
<h2>Regions (sigma from healthy; effect of making each one healthy)</h2>
<div class="card table-wrap"><table><tr><th>id</th><th>box (px)</th><th class="num">peak z</th><th class="num">P(answer) if healthy</th><th class="num">effect</th></tr>{rows}</table></div>
<h2>Reasoning</h2><pre>{html.escape(expl['narrative'])}</pre>
<p class="muted">Research software, not a medical device. "Healthy" means "like the reference scans used to build the atlas".</p>
</main></body></html>"""


def healthy_items(source, rng, n: int) -> list[ImageItem]:
    out = []
    for _ in range(n * 3):
        imgs = source.normal_images(rng)
        if imgs:
            out.append(imgs[0])
        if len(out) >= n:
            break
    return out


def evaluate_atlas(model, tok, source, atlas: "NormativeAtlas", n: int = 40, seed: int = 7, device=None) -> dict:
    """Label-free detection on a source's labelled examples: does 'deviation from healthy' separate
    abnormal from normal images, and (where masks exist) does it point at the abnormal pixels?"""
    from oncopattern.pipeline import auroc
    task = next((t for t in ("lesion", "present", "normal") if t in source.task_weights), None)
    if task is None:
        return {}
    rng = np.random.default_rng(seed)
    s, y, pz, py = [], [], [], []
    p = model.cfg.patch_size
    for _ in range(n):
        ex = source.sample(rng, task)
        abnormal = (ex.answer == "no") if task == "normal" else (ex.answer == "yes")
        dev = atlas.deviation(model, tok, ex.images[0], device)
        s.append(dev["score"])
        y.append(abnormal)
        mk = ex.meta.get("mask")
        if mk is not None and mk.any():
            g0, g1 = dev["z"].shape
            pz.append(dev["z"].ravel())
            py.append(mk[: g0 * p, : g1 * p].reshape(g0, p, g1, p).any((1, 3)).ravel())
    out = {"task_used_for_labels": task, "n": n, "image_auroc": auroc(s, y)}
    if pz:
        out["patch_auroc_vs_masks"] = auroc(np.concatenate(pz), np.concatenate(py))
    return out
