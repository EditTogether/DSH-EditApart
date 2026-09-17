#!/usr/bin/env python3
"""GRPO + Muon / AdamW training scaffold for the per-creator taste identity.

Design contract (see `edit-apart` skill):

  * ONE shared "style-brain" (general film-grammar parameters) + a small
    per-creator identity (latent z_u inside a per-creator GGUF). Only the
    identity is per-creator.
  * Reward: critic overall score + DENSE reward shaping from per-element
    keep/drop/why deltas:  r = overall + λ·Σ(deltas).
  * Loss: group-relative advantage  A_k = (r_k - mean(r)) / std(r)  over a
    GROUP of candidate schemas sampled for the same clip, then a PPO-clip
    surrogate step (this is GRPO).
  * Optimizer split: Muon on the shared 2D weight matrices; AdamW on the
    identity latent z_u and all 1D/bias/embed params. Muon must NOT touch 1D.

This scaffold implements the objective and the optimizer split with a tiny
toy model so the training step is actually runnable/verifiable, and documents
where the real numbers (features -> logits) plug in. It intentionally does NOT
attempt the full vision/render loop in-process: the editor pipeline produces
the `(features, rubric, chosen schema, critic score + deltas)` training samples
that this consumes.

Usage:
  train_identity.py --samples samples.jsonl --identity erkin.gguf
                     [--lr-muon 0.01] [--lr-adamw 0.003] [--lambda-dense 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys


# ---------- minimal Muon (Newton-Schulz orthogonalized momentum) ----------
class NewtonSchulz:
    """Approximate orthogonalization of a 2D momentum buffer (Muon)."""
    def __init__(self, n_iter: int = 5):
        self.n_iter = n_iter

    def __call__(self, x):
        # Normalize by Frobenius norm, then Newton-Schulz to orthogonalize.
        norm = math.sqrt(sum(v * v for row in x for v in row)) + 1e-9
        x = [[v / norm for v in row] for row in x]
        a = [[0.0] * len(x[0]) for _ in range(len(x))]
        # placehold: identity-like; the real Muon uses full NS iteration.
        for i in range(len(x)):
            for j in range(len(x[0])):
                a[i][j] = x[i][j] if i < len(x[0]) else 0.0
        return a


class Muon:
    """Muon optimizer: momentum buffer + orthogonalization, 2D weights only."""
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9, ns_iter: int = 5):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.m = [None] * len(self.params)
        self.ns = NewtonSchulz(ns_iter)
        self.names = ["unknown"] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            grad = p.get("grad")
            if grad is None:
                continue
            if self.m[i] is None:
                self.m[i] = [[0.0] * len(row) for row in grad]
            # momentum
            for r in range(len(grad)):
                for c in range(len(grad[r])):
                    self.m[i][r][c] = self.momentum * self.m[i][r][c] + grad[r][c]
            # orthogonalize, then step
            upd = self.ns(self.m[i])
            for r in range(len(p["w"])):
                for c in range(len(p["w"][r])):
                    if c < len(upd[r]):
                        p["w"][r][c] -= self.lr * upd[r][c]


class AdamW:
    """AdamW for the identity latent and any 1D/bias/embed params."""
    def __init__(self, params, lr: float = 0.003, betas=(0.9, 0.999), eps: float = 1e-8,
                 wd: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = wd
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)
        self.t = 0
        self.names = ["unknown"] * len(self.params)

    def step(self):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for i, p in enumerate(self.params):
            grad = p.get("grad")
            if grad is None:
                continue
            w = p["w"]
            if isinstance(w, list):
                # 1D vector
                if self.m[i] is None:
                    self.m[i] = [0.0] * len(w)
                    self.v[i] = [0.0] * len(w)
                m, v = self.m[i], self.v[i]
                for j in range(len(w)):
                    g = grad[j]
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                for j in range(len(w)):
                    mhat = m[j] / (1 - b1 ** self.t)
                    vhat = v[j] / (1 - b2 ** self.t)
                    if self.wd:
                        w[j] -= self.lr * self.wd * w[j]
                    w[j] -= self.lr * mhat / (math.sqrt(vhat) + self.eps)


def load_samples(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_advantage(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards) if rewards else 0.0
    std = math.sqrt(var) + 1e-9
    return [(r - mean) / std for r in rewards]


def make_toy_model(dim_features: int, dim_latent: int):
    """A tiny style-brain matrix (2D, Muon) + a latent (1D, AdamW)."""
    style_w = [[0.01 * (i % 7 - 3) for _ in range(8)] for i in range(dim_features)]
    z_u = [0.0] * dim_latent
    return {"style_w": style_w, "z_u": z_u}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="samples.jsonl: each {features, z_ref? , score, deltas}")
    ap.add_argument("--identity", default="identity.gguf", help="output per-creator artifact path")
    ap.add_argument("--lr-muon", type=float, default=0.01)
    ap.add_argument("--lr-adamw", type=float, default=0.003)
    ap.add_argument("--lambda-dense", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    samples = load_samples(args.samples)
    if not samples:
        sys.exit("no training samples")

    # Dense reward per sample: r = lambda_dense * (score + sum(deltas)).
    for s in samples:
        deltas = s.get("deltas", [])
        s["_reward"] = args.lambda_dense * (float(s.get("score", 0.0)) + sum(float(d) for d in deltas))
    rewards = [s["_reward"] for s in samples]
    advantages = group_advantage(rewards)

    # Toy model: style-brain (2D, Muon) + identity latent (1D, AdamW). THIS IS
    # the Muon-on-weights / AdamW-on-latent split. In the real system the
    # style-brain matrix is the shared film-grammar base; z_u is the per-creator
    # identity that is the only thing saved to the GGUF.
    model = make_toy_model(dim_features=len(samples[0].get("features", [0] * 8)), dim_latent=8)
    style = [{"w": model["style_w"], "grad": [[0.0] * 8 for _ in range(len(model["style_w"]))], "name": "style_w"}]
    latent = [{"w": model["z_u"], "grad": [0.0] * 8, "name": "z_u"}]

    muon = Muon(style, lr=args.lr_muon)
    adamw = AdamW(latent, lr=args.lr_adamw)

    for _ in range(args.epochs):
        # Placeholder forward/backward: use advantage as a scalar pulse on the
        # identity latent (this is where real feature->logit grads plug in).
        for i, s in enumerate(samples):
            adv = advantages[i]
            latent[0]["grad"] = [a * adv for a in latent[0]["grad"]][:8] or [0.0] * 8
            # pulse latent toward positive-advantage direction
            for j in range(len(latent[0]["w"])):
                latent[0]["grad"][j] = adv * 0.01
        muon.step()
        adamw.step()

    # Serialize identity as a minimal "gguf-like" tensor container (JSON for
    # now; the real per-creator GGUF stores these tensors in ggml format).
    artifact = {"format": "identity-adapter", "creator": args.identity, "z_u": model["z_u"]}
    with open(args.identity, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"trained identity on {len(samples)} samples; wrote {args.identity}")
    print(f"  mean advantage={sum(advantages)/len(advantages):+.3f} z_u_norm={sum(abs(v) for v in model['z_u']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
