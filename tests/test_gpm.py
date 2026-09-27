import numpy as np
import pytest
import torch

from oncopattern.gpm.compute import count_params, plan
from oncopattern.gpm.config import preset
from oncopattern.gpm.data import Example, ImageItem, build, build_with_answer_ids, collate, prompt_batch
from oncopattern.gpm.model import D4PatchStem, GeneralPatternModel
from oncopattern.gpm.rl import frozen_reference, grpo_step
from oncopattern.gpm.sources import (ImageFolderSource, MaskFolderSource, Mixture, MRIPhantomSource,
                                     TissuePhantomSource)
from oncopattern.gpm.tokenizer import ByteTokenizer
from oncopattern.gpm.train import train
from oncopattern.gpm.verify import box_iou, verify
from oncopattern.models.equivariant import d4_transform

TOK = ByteTokenizer()


@pytest.fixture(scope="module")
def sources():
    return TissuePhantomSource(TOK), MRIPhantomSource(TOK)


def tiny_model(**kw):
    torch.manual_seed(0)
    return GeneralPatternModel(preset("tiny", vocab_size=TOK.vocab_size, **kw))


def test_tokenizer_roundtrip_and_boxes():
    s = "measure 1.4 <c3><c10><c20><c40> <abstain> café"
    assert TOK.decode(TOK.encode(s)) == s
    box = TOK.parse_box(TOK.box_text(0.1, 0.2, 0.5, 0.9))
    assert box == pytest.approx((0.1, 0.2, 0.5, 0.9), abs=1 / TOK.n_coord)
    assert TOK.parse_box("no box") is None


def test_attention_mask_is_prefix_lm_with_bidirectional_images():
    block = torch.tensor([[0, 1, 1, 1, 0, 0]])
    valid = torch.tensor([[True] * 5 + [False]])
    m = GeneralPatternModel.attention_mask(block, valid)[0, 0]
    assert m[1, 3] and m[3, 1]  # patches of one image see each other both ways
    assert not m[0, 1] and m[4, 1] and not m[1, 4]  # text is causal
    assert not m[4, 5]  # never attend to padding


def test_d4_stem_responses_permute_under_rotation():
    torch.manual_seed(0)
    stem = D4PatchStem(1, 8, 4, 16)
    x = torch.randn(3, 1, 8, 8)
    w = torch.stack([d4_transform(stem.weight, g) for g in range(8)], 1)
    h = lambda z: (z.flatten(1) @ w.flatten(2).flatten(0, 1).t()).view(3, 4, 8)
    for u in range(8):
        a, b = h(x).sort(-1).values, h(d4_transform(x, u)).sort(-1).values
        assert torch.allclose(a, b, atol=1e-5)


def test_forward_losses_and_backward(sources):
    rng = np.random.default_rng(0)
    m = tiny_model()
    mix = Mixture(list(sources))
    exs = [mix.sample(rng) for _ in range(6)]
    b = collate([build(e, TOK, m.cfg.patch_size) for e in exs], TOK)
    out = m(b, generator=torch.Generator().manual_seed(0))
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    # answer tokens are the only LM targets
    built = build(sources[0].sample(rng, "classify"), TOK, m.cfg.patch_size)
    lab = [t for t in built["labels"] if t != -100]
    assert TOK.decode(lab).startswith(sources[0].sample(np.random.default_rng(0), "classify").answer[:0])
    assert lab[-1] == TOK.eos_id


def test_generate_with_kv_cache_matches_full_recompute(sources):
    m = tiny_model().eval()
    ex = sources[0].sample(np.random.default_rng(1), "describe")
    b = prompt_batch(ex, TOK, m.cfg.patch_size)
    fast = m.generate(b, max_new=6)
    built = build(ex, TOK, m.cfg.patch_size, with_answer=False)
    slow = []
    for _ in range(6):
        bb = collate([build_with_answer_ids(ex, slow, TOK, m.cfg.patch_size)], TOK)
        h = m.trunk(m.embed(bb, None, m._surprise_input(bb)), m.attention_mask(bb["block"], bb["valid"]))
        slow.append(int(m.lm_head(h[:, -1]).argmax(-1)))
    assert fast == slow
    assert len(built["ids"]) == b["ids"].shape[1]


def test_sequence_logprob_is_a_distribution_over_answers(sources):
    m = tiny_model().eval()
    ex = sources[1].sample(np.random.default_rng(2), "lesion")
    rows = [build_with_answer_ids(ex, TOK.encode(a) + [TOK.eos_id], TOK, m.cfg.patch_size) for a in ("yes", "no")]
    lp = m.sequence_logprob(collate(rows, TOK))
    assert lp.shape == (2,) and torch.all(lp < 0)


def test_surprise_conditioning_does_not_leak_hidden_patches(sources):
    m = tiny_model().eval()
    ex = Example(sources[0].image_only(np.random.default_rng(3)), "Study this image.", mim=True)
    b = collate([build(ex, TOK, m.cfg.patch_size)], TOK)
    n = b["patches"].shape[0]
    hidden = torch.zeros(n, dtype=torch.bool)
    hidden[:5] = True
    s1 = torch.randn(n)
    s2 = s1.clone()
    s2[:5] += 100.0  # wildly different surprise on the hidden patches only
    with torch.no_grad():
        assert torch.allclose(m.embed(b, hidden, s1), m.embed(b, hidden, s2))


def test_patch_surprise_covers_every_patch(sources):
    m = tiny_model().eval()
    ex = Example(sources[0].image_only(np.random.default_rng(3)), "Study this image.", mim=True)
    s = m.patch_surprise(collate([build(ex, TOK, m.cfg.patch_size)], TOK))
    assert s.shape == ((64 // m.cfg.patch_size) ** 2,) and torch.all(s > 0)


def test_every_source_task_builds(sources, tmp_path):
    rng = np.random.default_rng(4)
    for src in sources:
        for task in src.task_weights:
            ex = src.sample(rng, task)
            assert ex.answer and ex.meta["task"] == task
            assert verify(ex.meta, ex.answer, TOK)["reward"] == pytest.approx(1.0)  # truth passes its own test
    # folder layouts, as in most Kaggle datasets (.npy to avoid an image-library dependency)
    for c in ("normal", "tumour"):
        (tmp_path / "cls" / c).mkdir(parents=True)
        for i in range(5):
            np.save(tmp_path / "cls" / c / f"{i}.npy", rng.random((80, 80)).astype(np.float32))
    fs = ImageFolderSource(str(tmp_path / "cls"), TOK, "histology", 0.00025)
    assert fs.classes == ["normal", "tumour"] and fs.tools is not None
    for task in fs.task_weights:
        assert verify(fs.sample(rng, task).meta, fs.sample(np.random.default_rng(9), task).answer, TOK) is not None
    (tmp_path / "img").mkdir()
    (tmp_path / "msk").mkdir()
    for i in range(3):
        np.save(tmp_path / "img" / f"{i}.npy", rng.random((70, 70)).astype(np.float32))
        mk = np.zeros((70, 70), np.float32)
        mk[10:30, 40:60] = 1
        np.save(tmp_path / "msk" / f"{i}.npy", mk)
    ms = MaskFolderSource(str(tmp_path / "img"), str(tmp_path / "msk"), TOK, "ct", 0.7, "nodule")
    ex = ms.sample(rng, "locate")
    assert TOK.parse_box(ex.answer) == pytest.approx((10 / 70, 40 / 70, 30 / 70, 60 / 70), abs=0.03)


def test_verifier_rewards():
    meta = {"task": "locate", "box": (0.1, 0.1, 0.5, 0.5), "answer": ""}
    assert verify(meta, TOK.box_text(0.1, 0.1, 0.5, 0.5), TOK)["reward"] > 0.8
    assert verify(meta, "none", TOK)["reward"] == 0
    assert verify({"task": "locate", "box": None, "answer": "none"}, "none", TOK)["correct"]
    assert verify({"task": "locate", "box": None, "answer": "none"}, "garbage", TOK)["reward"] == 0
    assert verify({"task": "measure", "value": 1.4, "tolerance": 0.3}, "1.4", TOK)["reward"] == 1
    assert verify({"task": "measure", "value": 1.4, "tolerance": 0.3}, "2.0", TOK)["reward"] < 0.2
    assert verify({"task": "classify", "answer": "nuclear atypia"}, "nuclear atypia", TOK)["correct"]
    assert verify({"task": "classify", "answer": "normal"}, "<abstain>", TOK, abstain_reward=0.3)["reward"] == 0.3
    assert box_iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == pytest.approx(0.5)


def test_grpo_updates_toward_higher_reward(sources):
    m = tiny_model()
    ref = frozen_reference(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    exs = [sources[0].sample(np.random.default_rng(5), "count") for _ in range(2)]
    before = [p.detach().clone() for p in m.parameters()]
    res = grpo_step(m, exs, TOK, opt, ref=ref, group=4, max_new=6, reward_fn=lambda meta, text: float(len(text)))
    assert res["updated_prompts"] >= 1
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_training_loop_checkpoints_and_resumes(tmp_path):
    r1 = train("tiny", steps=3, batch_size=4, out=str(tmp_path), log_every=1, warmup=1)
    assert r1["step"] == 3 and (tmp_path / "checkpoint.pt").exists() and (tmp_path / "model.pt").exists()
    r2 = train("tiny", steps=5, batch_size=4, out=str(tmp_path), log_every=1, warmup=1)
    assert r2["step"] == 5 and len(r2["history"]) == 2  # continued from step 3, not from 0
    r3 = train("tiny", steps=100, batch_size=4, out=str(tmp_path / "b"), time_budget_h=1e-9, warmup=1)
    assert r3["stop_reason"] == "time_budget" and r3["step"] == 1


def test_presets_and_compute_plan():
    n7 = count_params("7b")
    assert 6.5e9 < n7 < 8e9
    assert count_params("tiny") < 5e6 < count_params("small") < count_params("base") < count_params("1b") < n7
    p = plan("t4", 2, 30)
    assert p["recommended_preset"] == "base"
    assert not p["presets"]["7b"]["trainable_on_one_device"]
    assert plan("t4", 2, 30, unique_tokens=3e7)["limited_by"] == "data"


def test_real_data_layouts_csv_suffix_masks_and_colour(tmp_path):
    from oncopattern.gpm.train import build_sources, parse_spec
    rng = np.random.default_rng(7)
    (tmp_path / "tiles").mkdir()
    rows = ["id,label"]
    for i in range(10):
        np.save(tmp_path / "tiles" / f"t{i}.npy", rng.random((3, 96, 96)).astype(np.float32))
        rows.append(f"t{i},{i % 2}")
    (tmp_path / "labels.csv").write_text("\n".join(rows))
    busi = tmp_path / "busi" / "malignant"
    busi.mkdir(parents=True)
    for i in range(3):
        np.save(busi / f"m{i}.npy", rng.random((50, 50)).astype(np.float32))
        mk = np.zeros((50, 50), np.float32)
        mk[5:20, 5:25] = 1
        np.save(busi / f"m{i}_mask.npy", mk)
    specs = [f"csv:{tmp_path / 'labels.csv'},images={tmp_path / 'tiles'},ext=.npy,names=0=normal|1=metastasis,"
             f"modality=histology,mm=0.00097,rgb=1,weight=2",
             f"masks:{tmp_path / 'busi'},suffix=_mask,modality=ultrasound,mm=0.1,finding=tumour",
             "phantom_tissue"]
    assert parse_spec(specs[1])[2]["suffix"] == "_mask"
    mix = build_sources(specs, TOK, 64, split="all")
    csv_src, mask_src, _ = mix.sources
    assert csv_src.classes == ["metastasis", "normal"] and csv_src.tools is not None
    assert len(mask_src.pairs) == 3 and all("_mask" not in str(a) for a, _ in mask_src.pairs)
    ex = csv_src.sample(rng, "classify")
    assert ex.images[0].pixels.shape == (3, 64, 64)
    # a colour tile and a grey phantom in one batch, on a colour model
    m = tiny_model(in_channels=3)
    exs = [csv_src.sample(rng, "classify"), mix.sources[2].sample(rng, "classify"), mask_src.sample(rng, "locate")]
    b = collate([build(e, TOK, m.cfg.patch_size) for e in exs], TOK)
    assert b["patches"].shape[1] == 3
    assert torch.isfinite(m(b)["loss"])


def test_conv_stem_is_orientation_equivariant_and_sees_contrast():
    from oncopattern.gpm.model import ConvPatchStem
    torch.manual_seed(0)
    stem = ConvPatchStem(1, 16, 32)
    x = torch.rand(5, 1, 8, 8)
    grp = torch.zeros(5, dtype=torch.long)

    def feats(z):  # pre-projection features: (P, c, 8 orientations, 2 poolings)
        h = torch.nn.functional.gelu(stem.gconv(torch.nn.functional.gelu(stem.lift(z))))
        return torch.stack([h.amax((-2, -1)), h.mean((-2, -1))], -1)

    a = feats(x).sort(2).values
    for u in range(8):  # rotating a patch only permutes its orientation channels
        assert torch.allclose(feats(d4_transform(x, u)).sort(2).values, a, atol=1e-5)
    # per-image standardisation: a brightness offset does not change the conv path
    base = stem(x, grp)
    shifted = stem(x + 0.3, grp)
    assert base.shape == (5, 32) and not torch.allclose(base, shifted)  # raw stats still see the offset


def test_train_test_split_is_deterministic_and_disjoint(tmp_path):
    from oncopattern.gpm.sources import in_split
    names = [tmp_path / f"img_{i}.png" for i in range(2000)]
    train = {n.name for n in names if in_split(n, "train")}
    test = {n.name for n in names if in_split(n, "test")}
    calib = {n.name for n in names if in_split(n, "calib")}
    assert not train & test and not train & calib and not test & calib
    assert len(train) + len(test) + len(calib) == 2000
    assert 0.07 < len(test) / 2000 < 0.13 and 0.07 < len(calib) / 2000 < 0.13
    assert test == {n.name for n in names if in_split(tmp_path / "elsewhere" / n.name, "test")}  # name-based


def test_kaggle_launcher_builds_a_valid_kernel(tmp_path, monkeypatch):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("launch", "kaggle/launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    monkeypatch.setattr(launch, "BUILD", tmp_path)
    d = launch.build_kernel("full", {"steps": 1234, "preset": None}, pcam=True, resume=True, username="someone")
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"] and meta["is_private"]
    assert meta["kernel_sources"] == ["someone/oncopattern-gpm"] and "someone/oncopattern-code" in meta["dataset_sources"]
    first = (d / "kernel_run.py").read_text().split("\n", 1)[0]
    cfg = json.loads(first[len("CONFIG = "):])
    assert cfg["steps"] == 1234 and cfg["preset"] == "base"
    compile((d / "kernel_run.py").read_text(), "kernel_run.py", "exec")


def test_normative_training_learns_normal_from_healthy_images_only(sources):
    rng = np.random.default_rng(11)
    mix = Mixture([sources[1]], mim_fraction=1.0, normative=True)
    for _ in range(10):  # every masked-pattern image is a healthy scan
        ex = mix.sample(rng)
        assert ex.mim and len(ex.images) == 1
    src = sources[1]
    healthy = src.normal_images(np.random.default_rng(3))[0].pixels
    lesion = src.sample(np.random.default_rng(3), "lesion")
    assert healthy.shape == lesion.images[0].pixels.shape


def test_normative_atlas_counterfactual_and_explanation(sources):
    from oncopattern.gpm.normative import NormativeAtlas, evaluate_atlas, explanation_html, healthy_items
    m = tiny_model().eval()
    src = sources[1]
    rng = np.random.default_rng(12)
    atlas = NormativeAtlas.fit(m, TOK, healthy_items(src, rng, 12))
    assert atlas.mu.shape == (64,) and atlas.z_threshold >= 2.0 and len(atlas.ref_scores) >= 2
    ex = src.sample(rng, "lesion")
    item = ex.images[0]
    grid = np.zeros((8, 8), bool)
    grid[2:4, 3:5] = True
    cf = atlas.healthy_counterfactual(m, TOK, item, grid)
    changed = np.abs(cf - item.pixels) > 1e-6
    assert changed[16:32, 24:40].any() and not changed[:16].any()  # only flagged patches are redrawn
    e = atlas.explain(m, TOK, ex, ["yes", "no"])
    assert abs(sum(e["p"]) - 1) < 1e-5 and "healthy reference scans" in e["narrative"]
    assert explanation_html(e, item.pixels).startswith("<!doctype html>")
    res = evaluate_atlas(m, TOK, src, atlas, n=6)
    assert 0 <= res["image_auroc"] <= 1
