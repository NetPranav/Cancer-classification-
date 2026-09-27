"""OncoPattern: the end-to-end pattern-level analysis pipeline.

Phases (see ROADMAP.md), in the order they must be run:

    pretrain(unlabelled)      Phase 3  label-free representation learning (optional)
    fit_normal(normals)       Phase 2  normality memory + normal reference statistics
    fit(images, labels)       Phase 4/5 concept evidence model, shrinkage prototypes, duplex reasoner
    calibrate(images, labels) Phase 7  fusion weights, temperature, conformal sets, risk-controlled abstention
    analyze(image)            Phase 6  prediction + highlighted regions + reasoning trace + narrative
    evaluate(images, labels)  Phase 8  metrics + failure ledger for the closed loop
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import rankdata

from oncopattern.calibration.conformal import (ConformalClassifier, SelectiveRiskController, fit_temperature,
                                               min_samples_for_guarantee, nll, softmax)
from oncopattern.data.phantoms import TISSUE_PATTERNS
from oncopattern.features.concepts import CONCEPTS, NormalReference, concept_maps
from oncopattern.loop.failures import CaseRecord, FailureLedger
from oncopattern.models.duplex import DuplexReasoner, spatial_encoding, train_duplex
from oncopattern.models.equivariant import PatternEncoder, d4_transform
from oncopattern.models.fewshot import ShrinkagePrototypes
from oncopattern.models.ssl import pretrain_ssl
from oncopattern.normality.memory import NormalityMemory
from oncopattern.reasoning.evidence import ConceptEvidenceModel, EvidenceItem
from oncopattern.reasoning.narrative import narrate

DESCRIPTIONS = dict(CONCEPTS, normality_deviation="local texture unlike any patch in the normal memory bank")
CONCEPT_NAMES = list(DESCRIPTIONS)
COMPONENTS = ("concept_evidence", "learned_prototypes", "duplex_stream")

DISCLAIMER = ("Research software. Output is a pattern-level finding on the supplied image, not a diagnosis, "
              "and is not validated for clinical use. A qualified clinician must review every case.")


@dataclass
class Config:
    classes: tuple[str, ...] = TISSUE_PATTERNS
    normal_class: str = "normal"
    image_size: int = 64
    token_grid: int = 8
    coreset_ratio: float = 0.1
    ssl_epochs: int = 4
    duplex_epochs: int = 60
    duplex_augment: bool = True
    halt_threshold: float = 0.9
    min_stream_steps: int = 4
    conformal_alpha: float = 0.1
    target_risk: float = 0.05
    min_concept_weight_share: float = 0.25
    delta: float = 0.1
    seed: int = 0


@dataclass
class Features:
    zmaps: dict
    concepts: np.ndarray
    embedding: np.ndarray
    tokens: np.ndarray
    pos: np.ndarray
    order: np.ndarray
    saliency: np.ndarray


@dataclass
class Region:
    region_id: str
    bbox: tuple[int, int, int, int]  # y0, x0, y1, x1 (exclusive)
    centre: tuple[float, float]
    area_px: int
    peak: float  # x typical-normal maximum
    dominant_concept: str
    concept_z: dict
    p_top_without: float | None = None  # counterfactual: P(top | concept evidence) if this region were normal


@dataclass
class Analysis:
    case_id: str
    classes: tuple[str, ...]
    probabilities: dict
    prediction: str
    answered: bool
    prediction_set: list
    component_probs: dict
    evidence: list  # EvidenceItem
    rival: str
    prior_db: float
    regions: list  # Region
    stream: list  # dicts
    tokens_read: int
    tokens_total: int
    highlight: np.ndarray
    concept_values: dict
    nearest_support: list
    selective_threshold: float
    narrative: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("highlight",)}
        d["evidence"] = [e.to_dict() if isinstance(e, EvidenceItem) else e for e in self.evidence]
        d["regions"] = [r.__dict__ for r in self.regions]
        d["classes"] = list(self.classes)
        return d


def auroc(scores, labels) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class OncoPattern:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        torch.manual_seed(self.cfg.seed)
        self.classes = tuple(self.cfg.classes)
        self.K = len(self.classes)
        self.encoder = PatternEncoder().eval()
        self.memory = NormalityMemory(coreset_ratio=self.cfg.coreset_ratio, seed=self.cfg.seed)
        self.reference = NormalReference()
        self.state = set()

    # ------------------------------------------------------------------ helpers
    def _require(self, *steps):
        missing = [s for s in steps if s not in self.state]
        if missing:
            raise RuntimeError(f"run {', '.join(missing)} first")

    def _check(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, np.float32)
        s = self.cfg.image_size
        if img.shape != (s, s):
            raise ValueError(f"expected a {s}x{s} image, got {img.shape}; see oncopattern.data.io.prepare")
        return img

    @torch.no_grad()
    def _dense(self, images: list[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(images)[:, None])
        return torch.cat([self.encoder(b) for b in x.split(64)])

    def _normality_map(self, dense_i: torch.Tensor) -> np.ndarray:
        c, h, w = dense_i.shape
        s = self.memory.score(dense_i.reshape(c, -1).T).reshape(h, w).numpy()
        up = ndi.zoom(s, self.cfg.image_size / h, order=1)
        return ndi.gaussian_filter(up, 1.5).astype(np.float32)

    def _raw_maps(self, images, dense=None):
        dense = self._dense(images) if dense is None else dense
        out = []
        for img, d in zip(images, dense):
            m = concept_maps(img)
            m["normality_deviation"] = self._normality_map(d)
            out.append(m)
        return out, dense

    def _featurize(self, images: list[np.ndarray]) -> list[Features]:
        images = [self._check(i) for i in images]
        maps, dense = self._raw_maps(images)
        g = self.cfg.token_grid
        feats = []
        for m, d in zip(maps, dense):
            z = self.reference.zmaps(m)
            peaks = self.reference.peaks(z)
            conc = np.array([peaks[c] / self.reference.scales[c] for c in CONCEPT_NAMES], np.float32)
            emb = torch.cat([d.mean((-2, -1)), d.amax((-2, -1))]).numpy()
            enc_tok = torch.nn.functional.adaptive_avg_pool2d(d[None], g)[0].reshape(d.shape[0], -1).T.numpy()
            zs = np.stack([np.clip(z[c] / self.reference.scales[c], -5, 50) for c in CONCEPT_NAMES])
            blk = self.cfg.image_size // g
            ztok = zs.reshape(len(CONCEPT_NAMES), g, blk, g, blk).max((2, 4)).reshape(len(CONCEPT_NAMES), -1).T
            tokens = np.concatenate([enc_tok, np.arcsinh(ztok)], 1).astype(np.float32)
            sal = np.clip(ztok, 0, None).max(1)
            order = np.argsort(-sal, kind="stable")
            rows, cols = torch.from_numpy(order // g), torch.from_numpy(order % g)
            pos = spatial_encoding(rows, cols, g).numpy()
            feats.append(Features(z, conc, emb, tokens[order], pos, order, sal[order]))
        return feats

    # ------------------------------------------------------------------ phases
    def pretrain(self, images: list[np.ndarray], epochs: int | None = None) -> list[float]:
        """Phase 3: label-free SSL on any images (normal, abnormal, unlabelled)."""
        if "fit_normal" in self.state:
            raise RuntimeError("pretrain must run before fit_normal (the memory bank stores encoder features)")
        x = torch.from_numpy(np.stack([self._check(i) for i in images])[:, None])
        hist = pretrain_ssl(self.encoder, x, epochs=epochs or self.cfg.ssl_epochs, seed=self.cfg.seed)
        self.state.add("pretrain")
        return hist

    def fit_normal(self, normals: list[np.ndarray], reference_fraction: float = 0.3) -> "OncoPattern":
        """Phase 2: store what normal looks like. Needs no abnormal examples."""
        normals = [self._check(i) for i in normals]
        n_ref = max(2, int(round(len(normals) * reference_fraction)))
        bank, ref = normals[n_ref:], normals[:n_ref]
        dense = self._dense(bank)
        self.memory.fit(dense.permute(0, 2, 3, 1).reshape(-1, dense.shape[1]))
        maps, _ = self._raw_maps(ref)
        self.reference.fit(maps)
        self.state.add("fit_normal")
        return self

    def fit(self, images: list[np.ndarray], labels: list[str]) -> "OncoPattern":
        """Phases 4-5: few-shot heads on a (small) labelled support set."""
        self._require("fit_normal")
        y = np.array([self.classes.index(l) for l in labels])
        feats = self._featurize(images)
        X = np.stack([f.concepts for f in feats])
        self.evidence = ConceptEvidenceModel(CONCEPT_NAMES).fit(X, y, self.K)
        self.protos = ShrinkagePrototypes().fit(np.stack([f.embedding for f in feats]), y, self.K)

        tok, pos, yy = [f.tokens for f in feats], [f.pos for f in feats], list(y)
        if self.cfg.duplex_augment:  # D4 views reorder the scan and move positions
            for g in range(1, 8):
                aug = self._featurize([d4_transform(torch.from_numpy(i), g).numpy().copy() for i in images])
                tok += [f.tokens for f in aug]
                pos += [f.pos for f in aug]
                yy += list(y)
        torch.manual_seed(self.cfg.seed)
        self.duplex = DuplexReasoner(tok[0].shape[1], self.K)
        self.duplex_loss = train_duplex(self.duplex, torch.from_numpy(np.stack(tok)), torch.from_numpy(np.stack(pos)),
                                        torch.tensor(yy), epochs=self.cfg.duplex_epochs, seed=self.cfg.seed)
        self.weights = np.array([1.0, 1.0, 1.0])
        self.comp_T = np.ones(3)
        self.T = 1.0
        self.state.add("fit")
        return self

    @torch.no_grad()
    def _duplex_full(self, f: Features):
        logits, halt = self.duplex(torch.from_numpy(f.tokens)[None], torch.from_numpy(f.pos)[None])
        return logits[0].numpy(), torch.sigmoid(halt[0]).numpy()

    @staticmethod
    def _exit_step(halt: np.ndarray, min_steps: int, threshold: float) -> int:
        """Index of the token at which the stream stops (the last one if it never halts)."""
        for t in range(min(min_steps, len(halt)) - 1, len(halt)):
            if halt[t] >= threshold:
                return t
        return len(halt) - 1

    def _components(self, f: Features, with_stream: bool = True):
        conc = self.evidence.logits(f.concepts[None])[0]
        proto = self.protos.logits(f.embedding[None])[0]
        stream, last = [], None
        if not with_stream:
            logits, halt = self._duplex_full(f)
            last = logits[self._exit_step(halt, self.cfg.min_stream_steps, self.cfg.halt_threshold)]
            return np.stack([conc, proto, last]), stream
        g = self.cfg.token_grid
        for t, logits, halt in self.duplex.stream(torch.from_numpy(f.tokens), torch.from_numpy(f.pos),
                                                  self.cfg.halt_threshold, self.cfg.min_stream_steps):
            p = softmax(logits.numpy())
            k = int(p.argmax())
            cell = int(f.order[t])
            stream.append({"step": t + 1, "patch_row": cell // g, "patch_col": cell % g,
                           "saliency": float(f.saliency[t]), "top": self.classes[k], "p_top": float(p[k]),
                           "halt": halt})
            last = logits.numpy()
        return np.stack([conc, proto, last]), stream

    def _fuse(self, comps: np.ndarray, weights=None, T=None) -> np.ndarray:
        w = self.weights if weights is None else weights
        z = comps / self.comp_T[:, None]
        logits = (w[:, None] * (z - z.max(-1, keepdims=True))).sum(0)
        return softmax(logits, self.T if T is None else T)

    def _calibrate_halting(self, feats: list[Features], max_disagreement: float = 0.02) -> dict:
        """CALM-style: choose the cheapest (min_steps, threshold) whose early answer agrees with
        the full read on >= 1 - max_disagreement of held-out cases; otherwise read everything."""
        full = [self._duplex_full(f) for f in feats]
        final = np.array([lg[-1].argmax() for lg, _ in full])
        T = len(full[0][1])
        best = {"min_steps": T, "threshold": 2.0, "agreement": 1.0, "mean_fraction_read": 1.0}
        for ms in (2, 4, 8, 16, 32):
            for th in (0.5, 0.7, 0.9, 0.97, 0.99):
                steps = np.array([self._exit_step(h, ms, th) for _, h in full])
                agree = float(np.mean([lg[s].argmax() for (lg, _), s in zip(full, steps)] == final))
                frac = float((steps + 1).mean() / T)
                if agree >= 1 - max_disagreement and frac < best["mean_fraction_read"]:
                    best = {"min_steps": ms, "threshold": th, "agreement": agree, "mean_fraction_read": frac}
        self.cfg.min_stream_steps, self.cfg.halt_threshold = best["min_steps"], best["threshold"]
        return best

    def calibrate(self, images: list[np.ndarray], labels: list[str]) -> dict:
        """Phase 7: split held-out labelled data in two halves. The first tunes fusion
        weights and temperature; the second fits the conformal and selective guarantees."""
        self._require("fit")
        y = np.array([self.classes.index(l) for l in labels])
        feats = self._featurize(images)
        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.permutation(len(y))
        tune, cal = idx[: len(y) // 2], idx[len(y) // 2:]
        halting = self._calibrate_halting([feats[i] for i in tune])
        comps = np.stack([self._components(f, with_stream=False)[0] for f in feats])

        # per-channel temperature first (naive-Bayes evidence is typically over-confident) ...
        self.comp_T = np.array([fit_temperature(comps[tune, i], y[tune]) for i in range(len(COMPONENTS))])
        # ... then fusion weights and a final temperature, all by held-out NLL
        # Faithfulness constraint: the interpretable concept channel always carries real weight,
        # so the deciban ledger explains part of the actual decision, not a bystander.
        best = (np.inf, None, None)
        grid = (0.0, 0.25, 0.5, 1.0, 2.0)
        for w in itertools.product(grid, repeat=3):
            w = np.array(w)
            if w[0] < self.cfg.min_concept_weight_share * w.sum() or w.sum() == 0:
                continue
            z = comps[tune] / self.comp_T[None, :, None]
            logits = (w[None, :, None] * (z - z.max(-1, keepdims=True))).sum(1)
            T = fit_temperature(logits, y[tune])
            loss = nll(logits, y[tune], T)
            if loss < best[0] - 1e-9:
                best = (loss, w, T)
        _, self.weights, self.T = best

        probs = np.stack([self._fuse(c) for c in comps[cal]])
        self.conformal = ConformalClassifier(self.cfg.conformal_alpha).fit(probs, y[cal])
        self.selective = SelectiveRiskController(self.cfg.target_risk, self.cfg.delta).fit(probs, y[cal])
        self.state.add("calibrate")
        return {"weights": dict(zip(COMPONENTS, self.weights.tolist())), "temperature": self.T,
                "channel_temperatures": dict(zip(COMPONENTS, self.comp_T.tolist())), "halting": halting,
                "tune_nll": best[0], "conformal_qhat": self.conformal.qhat,
                "selective": self.selective.result.__dict__,
                "n_needed_for_certificate": min_samples_for_guarantee(self.cfg.target_risk, self.cfg.delta),
                "n_calibration": int(len(cal))}

    # ------------------------------------------------------------------ inference
    def _regions(self, f: Features, top: int) -> list[Region]:
        comb = self.reference.combined(f.zmaps)
        lab, n = ndi.label(comb > self.reference.region_threshold)
        regions = []
        for i in range(1, n + 1):
            m = lab == i
            ys, xs = np.nonzero(m)
            cz = {c: float((f.zmaps[c][m] / self.reference.scales[c]).mean()) for c in CONCEPT_NAMES}
            # counterfactual: set this region's deviation back to normal and re-weigh the evidence
            zc = {c: np.where(ndi.binary_dilation(m, iterations=3), 0.0, v) for c, v in f.zmaps.items()}
            pk = self.reference.peaks(zc)
            x_wo = np.array([[pk[c] / self.reference.scales[c] for c in CONCEPT_NAMES]])
            p_wo = softmax(self.evidence.logits(x_wo)[0], self.comp_T[0])[top]
            regions.append(Region(f"R{i}", (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
                                  (float(ys.mean()), float(xs.mean())), int(m.sum()), float(comb[m].max()),
                                  max(cz, key=cz.get), cz, float(p_wo)))
        # rank by how much each region drives the decision (counterfactual drop), then by intensity
        base = softmax(self.evidence.logits(f.concepts[None])[0], self.comp_T[0])[top]
        regions.sort(key=lambda r: (-(base - r.p_top_without), -r.peak))
        for j, r in enumerate(regions, 1):
            r.region_id = f"R{j}"
        return regions[:8]

    def _analyze(self, f: Features, case_id: str) -> Analysis:
        comps, stream = self._components(f)
        p = self._fuse(comps)
        top = int(p.argmax())
        order = np.argsort(-p)
        rival = int(order[1])
        if top != self.classes.index(self.cfg.normal_class) and rival != self.classes.index(self.cfg.normal_class):
            # always also argue against "normal": it is the hypothesis a patient cares about most
            rival_for_ledger = self.classes.index(self.cfg.normal_class)
        else:
            rival_for_ledger = rival
        evidence = self.evidence.explain(f.concepts[None], top, rival_for_ledger, DESCRIPTIONS)
        pset = self.conformal.predict_set(p)[0]
        answered = bool(self.selective.accept(p)[0])
        nearest = [{"support_index": i, "label": self.classes[l], "distance": d}
                   for i, l, d in self.protos.nearest_support(f.embedding)]
        a = Analysis(
            case_id=case_id, classes=self.classes,
            probabilities={c: float(v) for c, v in zip(self.classes, p)},
            prediction=self.classes[top], answered=answered,
            prediction_set=[c for c, s in zip(self.classes, pset) if s],
            component_probs={n: {c: float(v) for c, v in zip(self.classes, softmax(comps[i], self.comp_T[i]))}
                              for i, n in enumerate(COMPONENTS)},
            evidence=evidence, rival=self.classes[rival_for_ledger],
            prior_db=self.evidence.prior_db(top, rival_for_ledger),
            regions=self._regions(f, top), stream=stream, tokens_read=len(stream),
            tokens_total=len(f.tokens), highlight=self.reference.combined(f.zmaps),
            concept_values={c: float(v) for c, v in zip(CONCEPT_NAMES, f.concepts)},
            nearest_support=nearest, selective_threshold=self.selective.result.threshold,
            extra={"weights": dict(zip(COMPONENTS, self.weights.tolist())), "temperature": self.T,
                   "region_threshold": self.reference.region_threshold},
        )
        a.narrative = narrate(a, self.cfg)
        return a

    def analyze(self, image: np.ndarray, case_id: str = "case") -> Analysis:
        self._require("calibrate")
        return self._analyze(self._featurize([image])[0], case_id)

    def analyze_batch(self, images, case_ids=None) -> list[Analysis]:
        self._require("calibrate")
        case_ids = case_ids or [f"case_{i:04d}" for i in range(len(images))]
        return [self._analyze(f, cid) for f, cid in zip(self._featurize(images), case_ids)]

    def evaluate(self, images, labels, metadata=None, masks=None, case_ids=None) -> tuple[dict, FailureLedger, list]:
        analyses = self.analyze_batch(images, case_ids)
        y = np.array([self.classes.index(l) for l in labels])
        P = np.stack([[a.probabilities[c] for c in self.classes] for a in analyses])
        pred = P.argmax(1)
        ans = np.array([a.answered for a in analyses])
        inset = np.array([labels[i] in a.prediction_set for i, a in enumerate(analyses)])
        normal = self.classes.index(self.cfg.normal_class)
        m = {
            "n": int(len(y)),
            "accuracy_all": float((pred == y).mean()),
            "coverage_answered": float(ans.mean()),
            "accuracy_on_answered": float((pred[ans] == y[ans]).mean()) if ans.any() else float("nan"),
            "conformal_coverage": float(inset.mean()),
            "conformal_target": 1 - self.cfg.conformal_alpha,
            "mean_set_size": float(np.mean([len(a.prediction_set) for a in analyses])),
            "auroc_abnormal_vs_normal": auroc(1 - P[:, normal], y != normal),
            "recall_per_class": {c: float((pred[y == k] == k).mean()) if (y == k).any() else float("nan")
                                 for k, c in enumerate(self.classes)},
            "mean_fraction_of_image_read": float(np.mean([a.tokens_read / a.tokens_total for a in analyses])),
        }
        for name, ci in zip(COMPONENTS, range(3)):
            cp = np.stack([[a.component_probs[name][c] for c in self.classes] for a in analyses])
            m[f"accuracy_{name}_alone"] = float((cp.argmax(1) == y).mean())
        if masks is not None:
            pix_s, pix_l, hits, n_ab = [], [], 0, 0
            for a, mk in zip(analyses, masks):
                if mk is None or not mk.any():
                    continue
                n_ab += 1
                pix_s.append(a.highlight.ravel())
                pix_l.append(mk.ravel())
                if a.regions:
                    y0, x0, y1, x1 = a.regions[0].bbox
                    hits += bool(mk[y0:y1, x0:x1].any())
            if n_ab:
                m["pixel_auroc"] = auroc(np.concatenate(pix_s), np.concatenate(pix_l))
                m["pointing_game"] = hits / n_ab
        ledger = FailureLedger()
        for i, a in enumerate(analyses):
            ledger.add(CaseRecord(a.case_id, labels[i], a.prediction, max(a.probabilities.values()), a.answered,
                                  bool(inset[i]), dict((metadata or [{}] * len(y))[i]), a.concept_values,
                                  list(a.concept_values.values())))
        return m, ledger, analyses

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "OncoPattern":
        return torch.load(path, weights_only=False)
