import numpy as np
import torch

from oncopattern.models.duplex import DuplexReasoner, spatial_encoding, train_duplex
from oncopattern.models.equivariant import COMPOSE, INVERSE, GroupConv, LiftingConv, PatternEncoder, d4_transform
from oncopattern.models.fewshot import ShrinkagePrototypes
from oncopattern.models.lora import add_lora, trainable_parameters
from oncopattern.models.ssl import pretrain_ssl
from oncopattern.normality.memory import NormalityMemory


def test_d4_tables_form_a_group():
    for i in range(8):
        assert COMPOSE[i][INVERSE[i]] == 0 and COMPOSE[0][i] == i
        for j in range(8):
            for k in range(8):
                assert COMPOSE[COMPOSE[i][j]][k] == COMPOSE[i][COMPOSE[j][k]]


def test_group_conv_is_exactly_equivariant():
    torch.manual_seed(0)
    lift, gc = LiftingConv(1, 3), GroupConv(3, 4)
    x = torch.randn(2, 1, 16, 16)
    out = gc(lift(x))
    for u in range(8):
        out_u = gc(lift(d4_transform(x, u)))
        expect = torch.stack([d4_transform(out[:, :, COMPOSE[INVERSE[u]][g]], u) for g in range(8)], 2)
        assert torch.allclose(out_u, expect, atol=1e-5)


def test_encoder_is_rotation_equivariant_and_embedding_invariant():
    torch.manual_seed(0)
    enc = PatternEncoder().eval()
    x = torch.randn(3, 1, 32, 32)
    with torch.no_grad():
        y, e = enc(x), enc.embed(x)
        for g in range(8):
            assert torch.allclose(enc(d4_transform(x, g)), d4_transform(y, g), atol=1e-5)
            assert torch.allclose(enc.embed(d4_transform(x, g)), e, atol=1e-5)


def test_ssl_runs_and_reduces_loss():
    torch.manual_seed(0)
    enc = PatternEncoder()
    hist = pretrain_ssl(enc, torch.rand(24, 1, 32, 32), epochs=4, batch_size=12)
    assert np.mean(hist[-2:]) < hist[0]


def test_memory_bank_flags_off_distribution_features():
    torch.manual_seed(0)
    normal = torch.randn(2000, 8)
    mem = NormalityMemory(coreset_ratio=0.1).fit(normal)
    assert mem.bank.shape[0] == 200
    s_in = mem.score(torch.randn(100, 8))
    s_out = mem.score(torch.randn(100, 8) + 4)
    assert s_out.min() > s_in.median()


def test_shrinkage_prototypes_few_shot():
    rng = np.random.default_rng(0)
    means = rng.normal(0, 3, (3, 5))
    Z = np.concatenate([means[k] + rng.normal(0, 1, (4, 5)) for k in range(3)])
    y = np.repeat(np.arange(3), 4)
    m = ShrinkagePrototypes().fit(Z, y, 3)
    Zt = np.concatenate([means[k] + rng.normal(0, 1, (50, 5)) for k in range(3)])
    acc = (m.logits(Zt).argmax(1) == np.repeat(np.arange(3), 50)).mean()
    assert acc > 0.85
    assert m.nearest_support(Z[0])[0][0] == 0


def test_lora_starts_as_noop_and_trains_few_parameters():
    torch.manual_seed(0)
    net = DuplexReasoner(10, 3)
    x, p = torch.randn(2, 5, 10), torch.randn(2, 5, 16)
    before = net(x, p)[0]
    total = sum(q.numel() for q in net.parameters())
    add_lora(net, r=2)
    assert torch.allclose(net(x, p)[0], before, atol=1e-6)
    assert 0 < trainable_parameters(net) < total / 4


def test_duplex_stream_matches_parallel_forward_and_halts():
    torch.manual_seed(0)
    net = DuplexReasoner(6, 3).eval()
    tok = torch.randn(12, 6)
    pos = spatial_encoding(torch.arange(12) // 4, torch.arange(12) % 4, 4)
    with torch.no_grad():
        full, halt = net(tok[None], pos[None])
    steps = list(net.stream(tok, pos, halt_threshold=2.0))  # never halts
    assert len(steps) == 12
    for t, logits, h in steps:
        assert torch.allclose(logits, full[0, t], atol=1e-5)
        assert abs(h - torch.sigmoid(halt[0, t]).item()) < 1e-5
    assert len(list(net.stream(tok, pos, halt_threshold=0.0, min_steps=3))) == 3


def test_duplex_learns_a_simple_rule():
    torch.manual_seed(0)
    n, T = 64, 8
    y = torch.randint(0, 2, (n,))
    tok = torch.randn(n, T, 4)
    tok[:, 0, 0] += 3 * (2 * y - 1)  # the answer is in the first (most salient) token
    pos = torch.zeros(n, T, 16)
    net = DuplexReasoner(4, 2, d=16, heads=2, layers=1)
    train_duplex(net, tok, pos, y, epochs=30)
    with torch.no_grad():
        logits, _ = net(tok, pos)
    assert (logits[:, 0].argmax(-1) == y).float().mean() > 0.9
