# OncoPattern roadmap

**Goal:** from minimal input (even a single scan or slide), detect *patterns*
of malignancy down to single-cell deviations, say **where** they are and
**why** (including the evidence that was weighed and discounted), give a
calibrated percentage, and **know when not to answer**. Every failure
becomes the next training target.

**Standing principles (the charter):**

1. *Never a silent error.* The model either answers within a certified error
   bound or refers the case. The pursuit of "100%" is a pursuit of lower
   certified error on answered cases, backed by numbers.
2. *Every explanation is traceable to a computed number.* No free-text
   hallucination.
3. *Label-light by design.* Learn normal from normals, representations from
   unlabelled data, and classes from a handful of examples.
4. *Honest scope.* Phantoms test the machinery. Only real, external,
   prospective data can support a clinical claim.

Status legend: ✅ implemented and tested · 🟡 partially implemented · ⬜ planned (needs data access or clinical partners)

---

## Phase 0: Foundations ✅
* Repository layout, pure-Python install (numpy, scipy, torch; CPU is enough).
* Charter above; disclaimer embedded in every output.
* Test suite, 29 tests: `pytest`.

**Exit criteria:** `python -m oncopattern demo --quick` runs end to end in under a minute on a laptop CPU. ✅

## Phase 1: Data layer ✅ / 🟡
* ✅ `data/catalogue.py`: machine-readable version of the dataset survey
  (38 entries: modality, organ, longitudinal, reports, access, licence),
  with discrepancies found while encoding recorded in `notes`.
* ✅ `data/phantoms.py`: procedural H&E-like tissue with *single-cell*
  anomalies (architectural disorder, nuclear atypia, hypercellularity) and
  head-MRI slices with growing lesions, all with exact ground-truth masks.
* ✅ `data/io.py`: PNG/JPEG, NumPy, DICOM (series), NIfTI loaders;
  normalisation; tiling; axial slicing.
* ⬜ Download adapters for TCIA/IDC (`idc-index`), Camelyon, PanNuke; stain
  normalisation (Macenko) for H&E; resampling to fixed mm or microns per pixel.

**Exit criteria:** one real cohort loaded through the same API as phantoms.

## Phase 2: Normality modelling ✅
* ✅ `physics/order.py`: nematic order parameter and misalignment (no training).
* ✅ `features/concepts.py`: 7 named concept maps, including loss of polarity,
  nuclear enlargement, hyperchromasia, cellularity, focal hyperintensity and
  asymmetry, plus robust normal-reference z-scoring.
* ✅ `normality/memory.py`: PatchCore memory bank with a k-center coreset.

**Exit criteria (phantoms):** abnormal-vs-normal AUROC > 0.95 and pixel AUROC > 0.9. **Met** (see README results).

## Phase 3: Data-efficient representation ✅
* ✅ D4-equivariant encoder, exactly equivariant (tested to 1e-5).
* ✅ Masked-view SSL pretraining (JEPA/SimSiam/VICReg hybrid).
* ✅ LoRA adapters for per-site or per-cancer adaptation.

**Exit criteria:** equivariance tests pass; SSL loss decreases. ✅

## Phase 4: Full-duplex streaming reasoner ✅
* ✅ Causal transformer with KV-cache streaming, identical to the parallel pass.
* ✅ Saliency-first scan order from label-free maps.
* ✅ CALM-style halting, with the stopping policy calibrated for ≥98% agreement with a full read.

**Exit criteria:** stream = parallel (tested); early exit keeps accuracy. ✅

## Phase 5: Few-shot heads ✅
* ✅ Shrinkage prototypes on learned embeddings (with nearest-case retrieval).
* ✅ Gaussian concept-evidence model with Bayesian shrinkage.

**Exit criteria:** above-chance classification with 8-12 examples per class. ✅

## Phase 6: Explanation ✅
* ✅ Deciban evidence ledger: supporting / counter-evidence outweighed /
  considered and set aside.
* ✅ Region extraction and a **counterfactual** per region ("if this region
  looked normal, P would fall from X to Y").
* ✅ Stream trace, nearest labelled cases, and a deterministic narrative.
* ✅ Self-contained HTML report (`reasoning/report.py`).

## Phase 7: Calibration and abstention ✅
* ✅ Per-channel and fused temperature scaling; fusion weights by held-out NLL
  with a faithfulness constraint.
* ✅ Split-conformal prediction sets.
* ✅ Learn-then-Test selective risk control, with the sample-size requirement made explicit.

## Phase 8: Closed failure loop ✅
* ✅ Failure ledger: errors, abstentions, conformal misses.
* ✅ Slice discovery with Wilson intervals; k-means++ failure clusters.
* ✅ Learning-curve power-law fit, giving samples needed or "unreachable with this data".
* ✅ Catalogue-driven acquisition recommendations; regression suite export.

**Operating loop:** evaluate → `failure_report.json` → acquire the
recommended data for flagged slices → retrain → the regression suite must pass → repeat.

## Phase 9: Longitudinal ✅ / 🟡
* ✅ Kalman local-linear trend, CUSUM change detection, doubling time.
* ✅ Demo: a lesion appears at month 12 and is tracked to month 24.
* ⬜ Deformable registration between timepoints (real scans are not pre-aligned).
* ⬜ Use RIDER test-retest scans to set per-modality noise floors (mu_0, sigma_0).

---

## Phase 10: Real-data validation ⬜ (next; needs data-use agreements)
Order matters: start where cell-level ground truth exists.

1. **PanNuke / Camelyon16** (pathology, cell and lesion masks): replace
   phantoms with real tiles. Fit the normal bank on tumour-free tiles, and
   measure lesion-level FROC and pixel AUROC.
2. **LIDC-IDRI / Lung-PET-CT-Dx** (radiology, nodule masks): 2.5D slices,
   then 3D (Phase 12).
3. **External validation** on a second site or scanner never seen in training
   (for example, train Camelyon16 Radboud, test Camelyon17 other centres).
4. **Subgroup audit** by scanner, stain, age, sex and ethnicity where
   available, using `FailureLedger.slices`.
5. **Reader study**: pathologists and radiologists rate whether highlighted
   regions and ledgers are correct and useful.

**Exit criteria:** external-site AUROC and a certified selective risk with a
reported coverage; calibration error (ECE) < 0.05.

## Phase 11: Future-risk (prognostic) modelling ⬜
"This patient may develop cancer" is a different task from "this image shows
a malignant pattern". It needs **outcome labels over time**. Precedent:
Sybil (lung cancer risk up to 6 years from one low-dose CT, trained on NLST)
and Mirai (breast cancer risk from mammograms).

* Relabel cohorts with "diagnosed within N years of this scan" (NLST has this).
* Replace the classification head with a discrete-time hazard head,
  P(cancer in year j | no cancer before j), and conformalise per horizon.
* The explanation machinery (regions, ledger, counterfactuals) is unchanged.

## Phase 12: Scale-up ⬜
* 3D volumes (group-equivariant 3D convolutions, or 2.5D with slice attention).
* Whole-slide images: tile → memory bank → attention multiple-instance learning;
  the duplex stream orders tiles.
* Multimodal: structured clinical variables and pathology reports as extra
  evidence items in the ledger.
* Optional LLM rephrasing of the narrative, *constrained to cite only
  ledger numbers*.

## Phase 13: Clinical translation ⬜
* Quality system (ISO 13485), software lifecycle (IEC 62304), risk management
  (ISO 14971), FDA SaMD / EU MDR pathway.
* Silent prospective deployment (model output hidden from clinicians) before any use.

---

## What is intentionally *not* claimed
* **100% accuracy.** No finite dataset can certify it. We certify an error
  bound on answered cases and refer the rest; see `docs/INNOVATIONS.md` §8 for
  the data required per bound.
* **Single-cell detection on MRI/CT.** A voxel is about 0.5-1 mm; a cell is
  about 10 µm. Single-cell patterns are only visible in microscopy (whole-slide
  images at about 0.25 µm per pixel). On radiology the same machinery detects
  tissue-level patterns.
* **Predicting cancer from a "normal" scan** without outcome-labelled
  longitudinal training data (Phase 11).
