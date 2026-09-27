"""Reinforcement learning from verifiable rewards (GRPO; Shao et al. 2024, from reasoning LLMs).

After supervised training, the model learns from checkable outcomes rather
than imitation. For each prompt, sample G answers, grade each with the
verifier, and push up the log-probability of answers that beat their
group's average:

    A_i = (r_i - mean(r)) / (std(r) + eps)
    L   = -1/G * sum_i A_i * log pi(a_i | x) / |a_i|  +  beta * KL(pi || pi_ref)

No learned reward model is needed. Rewards come from measurements, so they
cannot be flattered by fluent but wrong text, which is the failure mode
reported for accuracy-only medical VLM training. Abstention is paid a fixed
reward, so GRPO also learns *when* to decline.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from oncopattern.gpm.data import Example, build_with_answer_ids, collate, prompt_batch
from oncopattern.gpm.model import GeneralPatternModel
from oncopattern.gpm.tokenizer import ByteTokenizer
from oncopattern.gpm.verify import verify


def grpo_step(model: GeneralPatternModel, examples: list[Example], tok: ByteTokenizer, opt: torch.optim.Optimizer,
              ref: GeneralPatternModel | None = None, group: int = 4, temperature: float = 1.0, max_new: int = 16,
              beta: float = 0.02, abstain_reward: float = 0.3, seed: int = 0, device=None,
              reward_fn=None) -> dict:
    """One GRPO update over ``examples``. ``reward_fn(meta, text) -> float`` defaults to the verifier."""
    if reward_fn is None:
        def reward_fn(meta, text):
            return verify(meta, text, tok, abstain_reward)["reward"]
    gen = torch.Generator().manual_seed(seed)
    p = model.cfg.patch_size
    losses, rewards = [], []
    model.train()
    for ex in examples:
        model.eval()
        sampled = [model.generate(prompt_batch(ex, tok, p, device), max_new, temperature, gen, tok.eos_id)
                   for _ in range(group)]
        outs = [tok.decode(ids) for ids in sampled]
        model.train()
        r = np.array([reward_fn(ex.meta, o) for o in outs], float)
        rewards.extend(r.tolist())
        if r.std() < 1e-6:
            continue  # no learning signal when every sample scores the same
        adv = torch.tensor((r - r.mean()) / (r.std() + 1e-6), dtype=torch.float32)
        batch = collate([build_with_answer_ids(ex, ids, tok, p) for ids in sampled], tok, device)
        lens = (batch["labels"][:, 1:] != -100).sum(1).clamp(min=1).float()
        logp = model.sequence_logprob(batch)
        loss = -(adv.to(logp.device) * logp / lens).mean()
        if ref is not None and beta > 0:
            with torch.no_grad():
                ref_logp = ref.sequence_logprob(batch)
            ratio = (ref_logp - logp) / lens  # k3 estimator of KL(pi || ref)
            loss = loss + beta * (torch.exp(ratio) - ratio - 1).mean()
        losses.append(loss)
    if losses:
        opt.zero_grad()
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return {"mean_reward": float(np.mean(rewards)) if rewards else 0.0, "updated_prompts": len(losses)}


def frozen_reference(model: GeneralPatternModel) -> GeneralPatternModel:
    ref = copy.deepcopy(model).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    return ref
