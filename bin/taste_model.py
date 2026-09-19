#!/usr/bin/env python3
"""EditApart taste model — the real per-creator identity learner.

Design contract (see the `edit-apart` skill, "Taste model" section):

  * ONE shared "style-brain" (general film-grammar parameters) + a small
    per-creator identity (a latent `z_u`). Only the identity is per-creator, so
    a per-creator artifact is small and several creators share one style-brain.
  * The identity is ONE latent `z_u` injected at three points of a shared trunk:
    FiLM on the hidden layer (`h = (1 + tanh(W_g z)) * h1 + W_b z`), a latent gate
    on the output activations (`a * (1 + tanh(W_s z + b_s))`), and a per-creator
    linear readout over the features (`x . (z @ W_l)`). That is NONLINEAR in `z`
    and INTERPOLATABLE (the latent is continuous), which is what the design asks
    for; a single FiLM stage was measurably not enough for the latent to reorder
    candidates behind a frozen trunk.
  * Reward is dense + group-relative (GRPO): for a GROUP of K candidate schemas
    proposed for the same clip, `r_k = overall + lambda * sum(deltas)`,
    `A_k = (r_k - mean(r)) / std(r)`, and the update is a PPO-clip surrogate on
    the group-softmax selection policy. Utilities are standardised within the
    group first, so the logits stay responsive and the latent keeps the ability
    to reorder candidates. A pick logged with `chosen_by=creator|agent` is a
    REVEALED preference and becomes that group's top reward.
  * Optimizer split: Muon (Newton-Schulz orthogonalized momentum) on the shared
    2D weight matrices; AdamW on `z_u` and every 1D/bias parameter. Muon must
    never touch a 1D parameter.

What is real here (and what the previous scaffold faked):

  * `newton_schulz` is the actual quintic-iteration orthogonalization Muon uses,
    not an identity placeholder.
  * The forward pass is a real MLP with FiLM; the loss is a real GRPO surrogate
    (group-relative advantage + PPO-clip + entropy bonus) plus an auxiliary
    pairwise preference loss; gradients are analytic and checked against finite
    differences in `tests/test_taste_model.py`.
  * The artifact is a real GGUF v3 file (F32 tensors + metadata), not JSON:
    a shared `style_brain*.gguf` (the 2D/1D trunk) and a tiny per-creator
    `<creator>.gguf` carrying `z_u` and the style-brain digest it belongs to.

Honest boundary: the thing trained is the *taste scorer* over candidate edit
schemas (the selection policy), not the LLM that writes the schemas. Training
changes which candidate the loop picks; it does not fine-tune a language model.

CLI:
  taste_model.py train   --dataset D --identity ID [--style S] [...]
  taste_model.py score   --identity ID [--features JSON | --schema S --inventory I --rubric R]
  taste_model.py info    --identity ID
  taste_model.py init    --identity ID --style S [--creator NAME]
  taste_model.py features --schema S --inventory I --rubric R
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time

# ---------------------------------------------------------------------------
# feature layouts (pure stdlib: the engine logs these without needing numpy)
# ---------------------------------------------------------------------------
FEATURE_SPEC_VIDEO = "video/v1"

# Every feature is pre-normalized to roughly [0, 1.5] so the trunk starts in a
# sane range without any learned input scaling. Order IS the layout contract:
# an artifact records this list and refuses to load a vector that does not match.
VIDEO_FEATURES = [
    "log_n_segments",   # log1p(#segments) / log1p(60)
    "mean_shot_dur",    # mean kept shot duration / 6s
    "min_shot_dur",     # min kept shot duration / 6s
    "max_shot_dur",     # max kept shot duration / 6s
    "duration",         # total kept duration / 60s
    "target_ratio_err", # |total - target| / target
    "cpm",              # cuts per minute / 60
    "pace_band_hit",    # 1.0 if mean shot duration is inside the rubric pace band
    "frac_short",       # fraction of segments under the rubric minimum
    "mean_motion",      # mean source motion of kept shots / 12
    "frac_low_motion",  # fraction of kept shots the critic would call dead air
    "coverage",         # kept source seconds / source seconds
    "order_preserved",  # 1.0 = kept shots stay in source order
    "mean_lum",         # mean source luma p50 / 255
    "mean_rms",         # mean source RMS dB mapped from [-60, 0] to [0, 1]
    "bias",             # constant 1.0
]

FEATURE_SPECS = {FEATURE_SPEC_VIDEO: VIDEO_FEATURES}

# Pace bands, identical to the table the objective critic uses. Kept in sync on
# purpose: a feature must describe the same notion of "in band" the reward uses.
PACE_BANDS = {"slow": (2.0, 6.0), "medium": (1.0, 3.0), "brisk": (0.8, 2.5)}
DEFAULT_BAND = (0.8, 4.0)


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def features_video(schema: dict, inventory: dict, rubric: dict) -> dict:
    """Deterministic edit-schema features. No ffmpeg, no numpy, no I/O."""
    segs = [s for s in (schema.get("structure") or []) if isinstance(s, dict)]
    durs: list[float] = []
    for s in segs:
        trim = s.get("trim") or {}
        try:
            d = max(0.0, float(trim.get("out", 0.0)) - float(trim.get("in", 0.0)))
        except (TypeError, ValueError):
            d = 0.0
        durs.append(d)
    n = len(segs)
    total = sum(durs)
    mean_d = total / n if n else 0.0
    min_d = min(durs) if durs else 0.0
    max_d = max(durs) if durs else 0.0
    cpm = n / (total / 60.0) if total else 0.0

    inv_by = {shot.get("shot"): shot for shot in inventory.get("shots", [])
              if isinstance(shot, dict)}
    motions: list[float] = []
    lums: list[float] = []
    rmss: list[float] = []
    for s, d in zip(segs, durs):
        feat = (inv_by.get(s.get("shot"), {}) or {}).get("features") or {}
        if "motion" in feat:
            motions.append(float(feat["motion"]))
        if "lum_p50" in feat:
            lums.append(float(feat["lum_p50"]))
        if "rms_db" in feat:
            rmss.append(float(feat["rms_db"]))

    source_total = 0.0
    for shot in inventory.get("shots", []):
        try:
            source_total += float(shot.get("duration", 0.0))
        except (TypeError, ValueError):
            pass

    idx = [s.get("src_idx") for s in segs if isinstance(s.get("src_idx"), int)]
    if len(idx) >= 2:
        inversions = sum(1 for a, b in zip(idx, idx[1:]) if b < a)
        order_preserved = 1.0 - inversions / (len(idx) - 1)
    else:
        order_preserved = 1.0

    no_under = float(rubric.get("no_shot_under_s", 0.9) or 0.9)
    band = PACE_BANDS.get(rubric.get("pace"), DEFAULT_BAND)
    target = rubric.get("target_duration")
    feats = {
        "log_n_segments": _clip(math.log1p(n) / math.log1p(60.0), 0.0, 1.5),
        "mean_shot_dur": _clip(mean_d / 6.0, 0.0, 1.5),
        "min_shot_dur": _clip(min_d / 6.0, 0.0, 1.5),
        "max_shot_dur": _clip(max_d / 6.0, 0.0, 1.5),
        "duration": _clip(total / 60.0, 0.0, 2.0),
        "target_ratio_err": (_clip(abs(total - float(target)) / max(1.0, float(target)), 0.0, 1.0)
                             if target else 0.0),
        "cpm": _clip(cpm / 60.0, 0.0, 2.0),
        "pace_band_hit": 1.0 if (band[0] <= mean_d <= band[1]) else 0.0,
        "frac_short": (sum(1 for d in durs if d < no_under) / n) if n else 0.0,
        "mean_motion": _clip((sum(motions) / len(motions) if motions else 0.0) / 12.0, 0.0, 2.0),
        "frac_low_motion": ((sum(1 for m in motions if m < 1.0) / len(motions))
                            if motions else 0.0),
        "coverage": _clip(total / source_total, 0.0, 1.5) if source_total else 0.0,
        "order_preserved": _clip(order_preserved, 0.0, 1.0),
        "mean_lum": _clip((sum(lums) / len(lums) if lums else 0.0) / 255.0, 0.0, 1.5),
        "mean_rms": _clip(((sum(rmss) / len(rmss) if rmss else -60.0) + 60.0) / 60.0, 0.0, 1.5),
        "bias": 1.0,
    }
    return feats


def features_for(spec: str, schema: dict, inventory: dict, rubric: dict) -> dict:
    if spec == FEATURE_SPEC_VIDEO:
        return features_video(schema, inventory, rubric)
    raise ValueError(f"unknown feature spec {spec!r} (known: {sorted(FEATURE_SPECS)})")


def to_vector(feats: dict, spec: str) -> list[float]:
    """Ordered float vector for `spec`; a missing name is 0.0 (a logged older
    record must not make the whole dataset unreadable)."""
    names = FEATURE_SPECS.get(spec)
    if names is None:
        raise ValueError(f"unknown feature spec {spec!r}")
    out = []
    for name in names:
        try:
            out.append(float(feats.get(name, 0.0)))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the taste model needs numpy (it is installed with scenedetect; "
            "run setup.sh or `pip install 'numpy>=1.26,<2.0'`)") from exc
    return np


# ---------------------------------------------------------------------------
# GGUF v3 container (real file format: metadata KV + F32 tensors, 32B aligned)
# ---------------------------------------------------------------------------
# Bump when the parameter set changes. Artifacts carry it so an older style-brain
# is refused with guidance instead of silently loading the wrong trunk (or
# failing with a bare "missing tensor").
MODEL_LAYOUT = 2

GGUF_MAGIC = 0x46554747          # "GGUF" little-endian
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGML_TYPE_F32 = 0

# gguf metadata value-type ids
_GGUF_UINT8, _GGUF_INT8, _GGUF_UINT16, _GGUF_INT16 = 0, 1, 2, 3
_GGUF_UINT32, _GGUF_INT32, _GGUF_FLOAT32, _GGUF_BOOL = 4, 5, 6, 7
_GGUF_STRING, _GGUF_ARRAY, _GGUF_UINT64, _GGUF_INT64, _GGUF_FLOAT64 = 8, 9, 10, 11, 12

_GGUF_STRUCTS = {
    _GGUF_UINT8: "<B", _GGUF_INT8: "<b", _GGUF_UINT16: "<H", _GGUF_INT16: "<h",
    _GGUF_UINT32: "<I", _GGUF_INT32: "<i", _GGUF_FLOAT32: "<f", _GGUF_BOOL: "<B",
    _GGUF_UINT64: "<Q", _GGUF_INT64: "<q", _GGUF_FLOAT64: "<d",
}


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _gguf_type_of(value) -> int:
    if isinstance(value, bool):
        return _GGUF_BOOL
    if isinstance(value, str):
        return _GGUF_STRING
    if isinstance(value, int):
        return _GGUF_INT64
    if isinstance(value, float):
        return _GGUF_FLOAT32
    if isinstance(value, (list, tuple)):
        return _GGUF_ARRAY
    raise TypeError(f"cannot store {type(value).__name__} in GGUF metadata")


def _pack_value(fmt_type: int, value) -> bytes:
    if fmt_type == _GGUF_STRING:
        return _pack_string(str(value))
    if fmt_type == _GGUF_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if fmt_type == _GGUF_ARRAY:
        items = list(value)
        if items:
            elem_type = _gguf_type_of(items[0])
            for it in items:
                if _gguf_type_of(it) != elem_type:
                    raise TypeError("GGUF arrays must be homogeneous")
        else:
            elem_type = _GGUF_UINT8
        out = struct.pack("<I", elem_type) + struct.pack("<Q", len(items))
        for it in items:
            out += _pack_value(elem_type, it)
        return out
    return struct.pack(_GGUF_STRUCTS[fmt_type], value)


def write_gguf(path: str, metadata: dict, tensors: dict) -> dict:
    """Write a conformant GGUF v3 file. `tensors` values are numpy arrays; they
    are stored as F32 with GGUF's reversed-dimension convention, which makes a
    round-trip through any GGUF reader return the original numpy shape."""
    np = _numpy()
    meta = dict(metadata)
    meta.setdefault("general.alignment", GGUF_ALIGNMENT)
    alignment = int(meta["general.alignment"])

    blob = bytearray()
    blob += struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), len(meta))
    for key, value in meta.items():
        t = _gguf_type_of(value)
        blob += _pack_string(key) + struct.pack("<I", t) + _pack_value(t, value)

    prepared: list[tuple[str, "np.ndarray", int]] = []
    data = bytearray()
    for name, arr in tensors.items():
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        while len(data) % alignment:
            data += b"\x00"
        prepared.append((name, a, len(data)))
        data += a.tobytes()

    for name, a, offset in prepared:
        dims = tuple(int(d) for d in reversed(a.shape))
        blob += _pack_string(name)
        blob += struct.pack("<I", len(dims))
        for d in dims:
            blob += struct.pack("<Q", d)
        blob += struct.pack("<I", GGML_TYPE_F32)
        blob += struct.pack("<Q", offset)
    while len(blob) % alignment:
        blob += b"\x00"
    blob += data
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(blob))
    return {"path": path, "bytes": len(blob), "tensors": list(tensors),
            "metadata_keys": list(meta)}


def _read_value(fh, fmt_type: int):
    if fmt_type == _GGUF_STRING:
        (n,) = struct.unpack("<Q", fh.read(8))
        return fh.read(n).decode("utf-8")
    if fmt_type == _GGUF_ARRAY:
        (elem_type,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        return [_read_value(fh, elem_type) for _ in range(count)]
    if fmt_type == _GGUF_BOOL:
        return bool(struct.unpack("<B", fh.read(1))[0])
    return struct.unpack(_GGUF_STRUCTS[fmt_type], fh.read(struct.calcsize(_GGUF_STRUCTS[fmt_type])))[0]


def read_gguf(path: str) -> dict:
    """Read a GGUF v3 file written by `write_gguf` (also tolerates other GGUF
    writers' metadata so an artifact can be inspected with standard tooling)."""
    np = _numpy()
    with open(path, "rb") as fh:
        head = fh.read(24)
        if len(head) < 24:
            raise ValueError(f"{path}: not a GGUF file (header truncated: {len(head)} bytes)")
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", head)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (magic={magic:#x})")
        if version != GGUF_VERSION:
            raise ValueError(f"{path}: GGUF v{version} unsupported (need v{GGUF_VERSION})")
        meta = {}
        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", fh.read(8))
            key = fh.read(klen).decode("utf-8")
            (vtype,) = struct.unpack("<I", fh.read(4))
            meta[key] = _read_value(fh, vtype)
        infos = []
        for _ in range(n_tensors):
            (klen,) = struct.unpack("<Q", fh.read(8))
            tname = fh.read(klen).decode("utf-8")
            (n_dims,) = struct.unpack("<I", fh.read(4))
            dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(n_dims)]
            (ttype,) = struct.unpack("<I", fh.read(4))
            (offset,) = struct.unpack("<Q", fh.read(8))
            infos.append((tname, tuple(reversed(dims)), ttype, offset))
        alignment = int(meta.get("general.alignment", GGUF_ALIGNMENT))
        pos = fh.tell()
        if pos % alignment:
            fh.read(alignment - (pos % alignment))
        data_start = fh.tell()
        tensors = {}
        for tname, shape, ttype, offset in infos:
            if ttype != GGML_TYPE_F32:
                raise ValueError(f"{path}: tensor {tname} is ggml type {ttype}, only F32 supported")
            count = 1
            for d in shape:
                count *= int(d)
            fh.seek(data_start + offset)
            raw = fh.read(count * 4)
            tensors[tname] = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float64)
    return {"metadata": meta, "tensors": tensors}


# ---------------------------------------------------------------------------
# model: shared style-brain trunk + FiLM-modulated identity latent
# ---------------------------------------------------------------------------
def _param_specs(d_in: int, d_h: int, d_z: int) -> dict:
    """Name -> (shape, optimizer). 2D weights go to Muon; everything else is
    1D and goes to AdamW (including the per-creator latent)."""
    return {
        "W1": ((d_in, d_h), "muon"),
        "b1": ((d_h,), "adamw"),
        "Wg": ((d_z, d_h), "muon"),
        "bg": ((d_h,), "adamw"),
        "Wb": ((d_z, d_h), "muon"),
        "bb": ((d_h,), "adamw"),
        "W2": ((d_h, 1), "muon"),
        "b2": ((1,), "adamw"),
        # Second identity injection: the latent gates the OUTPUT activations and
        # also shifts the output bias. One FiLM stage on the hidden layer is not
        # enough for a frozen trunk to reorder candidates (measured: two
        # identities picked identically on 10/10 neutral briefs), because the
        # trunk's utilities saturate and a hidden-layer modulation barely moves
        # the ranking. Gating `a` before W2 makes z_u able to reorder directly —
        # still a single shared trunk plus one per-creator latent, and still
        # nonlinear + interpolatable.
        "Ws": ((d_z, d_h), "muon"),
        "bs": ((d_h,), "adamw"),
        "Wz": ((d_z, 1), "muon"),
        "bz": ((1,), "adamw"),
        # The identity's own linear readout over the edit features: w_u = z @ Wl
        # is a per-creator feature weighting (shared low-rank map Wl, per-creator
        # z_u). Without it the latent has to bend a saturated frozen trunk, which
        # measurably fails to learn a simple "long takes vs short takes"
        # preference (training accuracy 0.16 vs 1.00 for the opposite taste).
        "Wl": ((d_z, d_in), "muon"),
        "z_u": ((d_z,), "adamw"),
    }


class StyleBrain:
    """The shared trunk plus the per-creator latent, with real Muon/AdamW.

    `z_u` is the ONLY per-creator tensor; `W1..b2` are shared across creators
    and live in the style-brain artifact.
    """

    def __init__(self, d_in: int, d_h: int = 24, d_z: int = 16, seed: int = 0):
        np = _numpy()
        self.np = np
        self.d_in, self.d_h, self.d_z = int(d_in), int(d_h), int(d_z)
        self.spec = _param_specs(self.d_in, self.d_h, self.d_z)
        rng = np.random.default_rng(seed)
        self.p: dict[str, "np.ndarray"] = {}
        for name, (shape, _opt) in self.spec.items():
            if name == "z_u":
                self.p[name] = np.zeros(shape, dtype=np.float64)
            elif len(shape) == 2:
                scale = 1.0 / math.sqrt(max(1, shape[0]))
                self.p[name] = rng.normal(0.0, scale, shape)
            else:
                self.p[name] = np.zeros(shape, dtype=np.float64)
        # FiLM starts as identity: gamma = tanh(bg) = 0 ⇒ (1 + 0) * h1 + bb.
        self.muon = _Muon(self)
        self.adamw = _AdamW(self)

    # ---- forward / backward ------------------------------------------------
    def _trunk(self, X: "np.ndarray", z: "np.ndarray", cache: dict | None = None):
        """h1 -> FiLM(z) -> tanh -> bounded scalar utility.

        The utility is an UNBOUNDED ranking score. It is deliberately not
        squashed: a `tanh` output saturates under the group-relative margin
        objective, and once saturated the per-creator latent loses all authority
        over the ORDER (measured: two identities differing by +0.48 vs -0.84 on
        the same feature still chose identically on 12/12 briefs). Stability
        instead comes from standardising the utility inside each candidate group
        (see `policy_logits`), so the policy logits are bounded by construction
        and the latent can always reorder candidates.
        """
        np = self.np
        W1, b1, Wg, bg, Wb, bb, W2, b2, Ws, bs, Wz, bz, Wl = (self.p[k] for k in
                                                            ("W1", "b1", "Wg", "bg", "Wb", "bb",
                                                             "W2", "b2", "Ws", "bs", "Wz", "bz",
                                                             "Wl"))
        h1 = X @ W1 + b1
        pre_g = z @ Wg + bg
        g = np.tanh(pre_g)
        be = z @ Wb + bb
        h = (1.0 + g) * h1 + be
        a = np.tanh(h)
        pre_s = z @ Ws + bs
        gate = 1.0 + np.tanh(pre_s)                 # per-creator output gate
        lin = (X @ (z @ Wl))[:, None]               # per-creator feature readout (N,1)
        u = (a * gate) @ W2 + b2 + (z @ Wz) + bz + lin
        if cache is not None:
            cache.update(dict(X=X, z=z, h1=h1, pre_g=pre_g, g=g, be=be, h=h, a=a,
                              pre_s=pre_s, gate=gate, lin=lin, u=u))
        return u

    def utilities(self, X, z=None) -> "np.ndarray":
        """(N,) taste utilities for a batch of candidate feature vectors."""
        np = self.np
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.d_in:
            raise ValueError(f"feature vector has {X.shape[1]} dims, model wants {self.d_in}")
        if z is None:
            z = self.p["z_u"]
        return self._trunk(X, np.asarray(z, dtype=np.float64)).reshape(-1)

    def backward(self, dU: "np.ndarray", cache: dict) -> dict:
        """Analytic gradients from d(loss)/d(utilities). Verified against finite
        differences in tests/test_taste_model.py."""
        np = self.np
        X, z, h1, g, h, a, gate, u = (cache[k] for k in ("X", "z", "h1", "g", "h", "a",
                                                         "gate", "u"))
        W1, Wg, Wb, W2, Ws, Wz, Wl = (self.p[k] for k in ("W1", "Wg", "Wb", "W2", "Ws", "Wz",
                                                          "Wl"))
        dU = np.asarray(dU, dtype=np.float64).reshape(-1, 1)
        grads: dict[str, "np.ndarray"] = {}

        dz2 = dU                                      # linear (unbounded) ranking score
        grads["W2"] = (a * gate).T @ dz2
        grads["b2"] = dz2.sum(axis=0)
        grads["Wz"] = np.outer(z, dz2.sum(axis=0))
        grads["bz"] = dz2.sum(axis=0)
        grads["Wl"] = np.outer(z, (dz2 * X).sum(axis=0))
        d_ag = dz2 @ W2.T                             # d(a * gate)
        da = d_ag * gate
        d_gate = d_ag * a
        d_pre_s = d_gate * (1.0 - np.tanh(cache["pre_s"]) ** 2)
        grads["Ws"] = np.outer(z, d_pre_s.sum(axis=0))
        grads["bs"] = d_pre_s.sum(axis=0)
        dh = da * (1.0 - a ** 2)

        dW1_core = dh * (1.0 + g)
        grads["W1"] = X.T @ dW1_core
        grads["b1"] = dW1_core.sum(axis=0)

        # FiLM: g and be depend on z only, so their batch contributions add up.
        d_pre_g = dh * h1 * (1.0 - g ** 2)          # dL/d(pre_g), (N, d_h)
        grads["Wg"] = np.outer(z, d_pre_g.sum(axis=0))
        grads["bg"] = d_pre_g.sum(axis=0)
        grads["Wb"] = np.outer(z, dh.sum(axis=0))
        grads["bb"] = dh.sum(axis=0)
        grads["z_u"] = ((d_pre_g.sum(axis=0) @ Wg.T) + (dh.sum(axis=0) @ Wb.T)
                        + (d_pre_s.sum(axis=0) @ Ws.T) + (dz2.sum(axis=0) @ Wz.T)
                        + ((dz2 * X).sum(axis=0) @ Wl.T))
        return grads

    def zero_grad(self):
        self._grads = {}

    def add_grads(self, grads: dict):
        acc = getattr(self, "_grads", None)
        if acc is None:
            acc = self._grads = {}
        for k, v in grads.items():
            acc[k] = v if k not in acc else acc[k] + v

    def grad_norm(self) -> float:
        np = self.np
        grads = getattr(self, "_grads", None) or {}
        if not grads:
            return 0.0
        return float(np.sqrt(sum(float((g ** 2).sum()) for g in grads.values())))

    def apply_grads(self, max_norm: float | None = 1.0) -> dict:
        """Clip the accumulated gradient by global norm, then step Muon + AdamW.

        Clipping is not decoration here: the PPO-family objective contains an
        unbounded `rho * advantage` term, and without it the logits — and the
        reported loss — run away (measured: loss ~4e9 within 10 epochs).
        """
        grads = getattr(self, "_grads", None) or {}
        raw = self.grad_norm()
        clipped = False
        if max_norm and raw > max_norm:
            scale = max_norm / (raw + 1e-12)
            grads = {k: v * scale for k, v in grads.items()}
            clipped = True
        self.muon.step(grads)
        self.adamw.step(grads)
        self._grads = {}
        return {"grad_norm": raw, "clipped": clipped}

    # ---- artifacts ---------------------------------------------------------
    def shared_tensors(self) -> dict:
        return {k: self.p[k] for k in self.spec if k != "z_u"}

    def shared_digest(self) -> str:
        h = hashlib.sha256()
        for name in sorted(self.shared_tensors()):
            h.update(name.encode())
            h.update(self.np.ascontiguousarray(self.shared_tensors()[name], dtype=self.np.float32).tobytes())
        return h.hexdigest()

    def load_shared(self, path: str) -> dict:
        art = read_gguf(path)
        meta = art["metadata"]
        if meta.get("editapart.kind") != "style_brain":
            raise ValueError(f"{path}: not a style_brain artifact")
        layout = meta.get("editapart.model_layout")
        if layout != MODEL_LAYOUT:
            raise ValueError(
                f"{path}: style-brain model layout {layout!r} does not match this build "
                f"({MODEL_LAYOUT}); retrain the shared trunk (train_identity without "
                f"--freeze-style) to upgrade it")
        d_in, d_h, d_z = (int(meta["editapart.d_in"]), int(meta["editapart.d_h"]),
                          int(meta["editapart.d_z"]))
        if (d_in, d_h, d_z) != (self.d_in, self.d_h, self.d_z):
            raise ValueError(f"{path}: dims {(d_in, d_h, d_z)} != model "
                             f"{(self.d_in, self.d_h, self.d_z)}")
        for name in self.shared_tensors():
            if name not in art["tensors"]:
                raise ValueError(f"{path}: missing tensor {name}")
            self.p[name] = self.np.asarray(art["tensors"][name], dtype=self.np.float64)
        return meta


# ---------------------------------------------------------------------------
# optimizers — Muon (real Newton-Schulz) on 2D, AdamW on 1D
# ---------------------------------------------------------------------------
def newton_schulz(G, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton-Schulz orthogonalization — the transform Muon applies to a
    momentum buffer.

    Coefficients and ordering follow the reference implementation
    (KellerJordan/Muon, `zeropower_via_newtonschulz5`, coefficients
    `(3.4445, -4.7750, 2.0315)`): transpose so rows <= cols, normalize by the
    Frobenius norm so the spectral norm starts at most 1, then iterate
    `X <- a*X + (b*A + c*A^2) @ X` with `A = X @ X.T`.

    IMPORTANT (and verified in the test suite): this is an APPROXIMATE
    orthogonalization. The reference explicitly documents that it does not
    converge to `U V^T` on the whole interval — the result is `U S' V^T` with
    singular values spread over roughly [0.5, 1.5]. A test asserting exact
    orthogonality would be testing the wrong contract.
    """
    np = _numpy()
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("Muon's Newton-Schulz applies to 2D matrices only")
    X = G / (np.linalg.norm(G) + eps)
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class _Muon:
    """Momentum + orthogonalization over 2D parameters only. 1D parameters in
    the same dict are ignored here and handled by AdamW.

    Mirrors the reference `muon_update`: `m <- beta*m + (1-beta)*g`, then the
    nesterov update `u = (1-beta)*g + beta*m`, then Newton-Schulz on `u`, then
    the `sqrt(max(1, rows/cols))` scale, then `p -= lr * u`.
    """

    def __init__(self, model: StyleBrain, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0, ns_steps: int = 5, nesterov: bool = True):
        self.model = model
        self.np = model.np
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.buf: dict[str, "np.ndarray"] = {}
        self.touched_1d: list[str] = []          # invariant witness: must stay empty

    def targets(self):
        return [n for n, (shape, opt) in self.model.spec.items()
                if opt == "muon" and len(shape) == 2]

    def step(self, grads: dict):
        np = self.np
        for name in self.targets():
            g = grads.get(name)
            if g is None:
                continue
            if g.ndim != 2:
                self.touched_1d.append(name)
                raise AssertionError(f"Muon received a non-2D gradient for {name}")
            if name not in self.buf:
                self.buf[name] = np.zeros_like(g)
            beta = self.momentum
            self.buf[name] = beta * self.buf[name] + (1.0 - beta) * g
            update = (1.0 - beta) * g + beta * self.buf[name] if self.nesterov else self.buf[name]
            ns = newton_schulz(update, self.ns_steps)
            rows, cols = self.model.p[name].shape
            scale = math.sqrt(max(1.0, rows / cols))
            if self.weight_decay:
                self.model.p[name] = self.model.p[name] * (1.0 - self.lr * self.weight_decay)
            self.model.p[name] = self.model.p[name] - self.lr * scale * ns


class _AdamW:
    """AdamW over 1D parameters only — the per-creator latent and all biases."""

    def __init__(self, model: StyleBrain, lr: float = 0.01, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.model = model
        self.np = model.np
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: dict[str, "np.ndarray"] = {}
        self.v: dict[str, "np.ndarray"] = {}
        self.t = 0
        self.touched_2d: list[str] = []          # invariant witness: must stay empty

    def targets(self):
        return [n for n, (shape, opt) in self.model.spec.items()
                if opt == "adamw" and len(shape) == 1]

    def step(self, grads: dict):
        np = self.np
        self.t += 1
        for name in self.targets():
            g = grads.get(name)
            if g is None:
                continue
            if g.ndim != 1:
                self.touched_2d.append(name)
                raise AssertionError(f"AdamW received a 2D gradient for {name}")
            if name not in self.m:
                self.m[name] = np.zeros_like(g)
                self.v[name] = np.zeros_like(g)
            m, v = self.m[name], self.v[name]
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            self.m[name], self.v[name] = m, v
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            w = self.model.p[name]
            if self.weight_decay:
                w = w * (1.0 - self.lr * self.weight_decay)
            w = w - self.lr * mhat / (np.sqrt(vhat) + self.eps)
            self.model.p[name] = w


# ---------------------------------------------------------------------------
# GRPO objective
# ---------------------------------------------------------------------------
def group_advantage(rewards, eps: float = 1e-9):
    np = _numpy()
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return r
    std = r.std()
    if std < eps:
        return r - r.mean()          # zero signal, but a well-defined zero
    return (r - r.mean()) / (std + eps)


def log_softmax(s):
    np = _numpy()
    m = s.max()
    return s - m - math.log(float(np.exp(s - m).sum()))


def softmax(s):
    return _numpy().exp(log_softmax(s))


def policy_logits(model: StyleBrain, X, temperature: float, eps: float = 1e-8) -> dict:
    """Standardise the utilities WITHIN the candidate group, then scale.

    `s_k = ((u_k - mean(u)) / std(u)) / temperature`. Only the ordering inside a
    group is used for selection, and standardising keeps the logits in a
    responsive range no matter how confident the trunk becomes — this is what
    stops the softmax from going one-hot and killing the gradient the per-creator
    latent needs in order to reorder candidates.
    """
    np = model.np
    u = model.utilities(X)
    mean = float(u.mean())
    std = float(u.std()) + eps
    v = (u - mean) / std
    return {"u": u, "v": v, "s": v / temperature, "mean": mean, "std": std}


def logits_to_utility_grads(dloss_ds, v, std: float, temperature: float):
    """d(loss)/d(utility) given d(loss)/d(standardised logits).

    s_k = (u_k - m)/(sigma*T)  ⇒
    dloss/du_j = [ g_j - mean(g) - v_j * mean(g*v) ] / (sigma*T)
    """
    np = _numpy()
    g = np.asarray(dloss_ds, dtype=np.float64)
    return (g - g.mean() - v * float((g * v).mean())) / (std * temperature)


def grpo_loss_and_grads(model: StyleBrain, groups: list[dict],
                        old_logits: list, temperature: float = 0.5,
                        clip_eps: float = 0.2, entropy_coef: float = 0.01,
                        pref_coef: float = 0.5, model_grads_out: dict | None = None):
    """Dense group-relative REINFORCE-with-clip objective + pairwise preference.

    `groups[i]` = {"X": (N,d_in) array, "A": (N,) advantages, "chosen": idx|None}
    `old_logits[i]` = the logits that were current when the ratio baseline was
    taken (standard PPO: the ratio baseline is frozen for the epoch).
    """
    np = model.np
    total = 0.0
    n_groups = 0
    clip_hits = 0
    clip_total = 0
    mean_rho_dev = 0.0
    grad_acc: dict[str, "np.ndarray"] = {}

    for gi, grp in enumerate(groups):
        X = grp["X"]
        if X.shape[0] < 2:
            continue                      # a group of one has no relative signal
        pol = policy_logits(model, X, temperature)
        u, v, s = pol["u"], pol["v"], pol["s"]
        lpi = log_softmax(s)
        lpi_old = old_logits[gi]
        rho = np.exp(lpi - lpi_old)
        A = grp["A"]
        unclipped = rho * A
        clipped = np.clip(rho, 1.0 - clip_eps, 1.0 + clip_eps) * A
        take_unclipped = unclipped <= clipped
        obj = np.where(take_unclipped, unclipped, clipped)
        clip_hits += int((~take_unclipped).sum())
        clip_total += int(rho.size)
        mean_rho_dev += float(np.abs(rho - 1.0).mean())

        pi = softmax(s)
        H = -float((pi * lpi).sum())
        # dObjective/ds : -(1/N) * [mask_m A_m rho_m - pi_m * sum_n mask_n A_n rho_n]
        mask = take_unclipped.astype(np.float64) * A * rho
        sum_mask = float(mask.sum())
        dloss_ds = -(mask - pi * sum_mask) / (mask.size)
        dloss_ds += -entropy_coef * (-pi * (lpi + H))

        # Auxiliary pairwise preference: the candidate the loop actually chose
        # should outrank the alternatives (this is where the render+critic
        # outcome enters the model even when full-group rewards are sparse).
        pref_loss = 0.0
        chosen = grp.get("chosen")
        if pref_coef and chosen is not None and X.shape[0] > 1:
            others = [j for j in range(X.shape[0]) if j != chosen]
            if others:
                d = s[chosen] - s[others]
                sigma = 1.0 / (1.0 + np.exp(-d))
                # -log sigmoid(d) averaged over the alternatives
                pref_loss = -pref_coef * float(np.log(sigma).mean())
                dloss_ds[chosen] += -pref_coef * float((1.0 - sigma).mean())
                contrib = pref_coef * (1.0 - sigma) / len(others)
                for j, c in zip(others, contrib):
                    dloss_ds[j] += float(c)

        loss_i = float(np.mean(obj)) if obj.size else 0.0
        total += -loss_i - entropy_coef * H + pref_loss
        n_groups += 1

        du = logits_to_utility_grads(dloss_ds, v, pol["std"], temperature).reshape(-1, 1)
        cache: dict = {}
        model._trunk(X, model.p["z_u"], cache)
        grads = model.backward(du, cache)
        for k, v in grads.items():
            grad_acc[k] = v if k not in grad_acc else grad_acc[k] + v

    if model_grads_out is not None:
        if n_groups:
            for k in grad_acc:
                grad_acc[k] = grad_acc[k] / n_groups
        model_grads_out.clear()
        model_grads_out.update(grad_acc)

    return {
        "loss": (total / n_groups) if n_groups else 0.0,
        "groups": n_groups,
        "clip_fraction": (clip_hits / clip_total) if clip_total else 0.0,
        "mean_abs_rho_minus_1": (mean_rho_dev / n_groups) if n_groups else 0.0,
    }


# ---------------------------------------------------------------------------
# dataset folding: the loop writes append-only JSONL, training reads groups
# ---------------------------------------------------------------------------
def load_groups(path: str) -> tuple[list[dict], dict]:
    """Fold the loop's append-only log into training groups.

    Records:
      {"kind":"group","group_id":G,"feature_spec":"video/v1",
       "candidates":[{"features":{...},"reward_obj":r,"overall_obj":o,"dense_obj":d,
                      "knobs":{...}}],"chosen":i}
      {"kind":"reward","group_id":G,"candidate":j,"reward":r,"subjective":s}

    A later reward record overrides that candidate's reward (the render+vision
    leg refines the cheap objective reward), last write wins. When the separate
    overall/dense components are present the reward is re-derived at training
    time as `overall + lambda_dense * dense`, which is what makes `lambda_dense`
    a real knob rather than a no-op.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    stats = {"records": 0, "legacy_singletons": 0, "reward_records": 0, "bad_lines": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["records"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_lines"] += 1
                continue
            kind = rec.get("kind")
            if kind == "reward":
                g = groups.get(str(rec.get("group_id")))
                if not g:
                    continue
                idx = int(rec.get("candidate", 0))
                if 0 <= idx < len(g["candidates"]):
                    c = g["candidates"][idx]
                    g["chosen_by"] = str(rec.get("chosen_by", "critic"))
                    c["reward"] = float(rec.get("reward", 0.0))
                    if rec.get("overall_obj") is not None:
                        c["overall"] = float(rec["overall_obj"])
                    if rec.get("dense_obj") is not None:
                        c["dense"] = float(rec["dense_obj"])
                    c["reward_source"] = "critic"
                    if rec.get("subjective") is not None:
                        c["subjective"] = float(rec["subjective"])
                    g["chosen"] = idx
                stats["reward_records"] += 1
                continue
            if kind == "group":
                gid = str(rec.get("group_id") or f"g{len(order)}")
                g = {"group_id": gid, "clip_id": rec.get("clip_id"),
                     "feature_spec": rec.get("feature_spec", FEATURE_SPEC_VIDEO),
                     "chosen": None, "chosen_by": None, "candidates": []}
                for c in rec.get("candidates", []):
                    g["candidates"].append({
                        "features": c.get("features") or {},
                        "reward": float(c.get("reward_obj", c.get("reward", 0.0))),
                        "overall": (None if c.get("overall_obj") is None
                                    else float(c["overall_obj"])),
                        "dense": (None if c.get("dense_obj") is None
                                  else float(c["dense_obj"])),
                        "reward_source": "objective",
                        "knobs": c.get("knobs") or {},
                    })
                ch = rec.get("chosen")
                g["chosen"] = int(ch) if ch is not None else None
                groups[gid] = g
                order.append(gid)
                continue
            # legacy flat record from the old scaffold: {features, score, deltas}
            if "features" in rec:
                feats = rec["features"]
                if isinstance(feats, list):
                    spec = FEATURE_SPEC_VIDEO
                    feats = {n: v for n, v in zip(FEATURE_SPECS[spec], feats)}
                else:
                    spec = FEATURE_SPEC_VIDEO
                deltas = rec.get("deltas", []) or []
                reward = float(rec.get("score", 0.0)) + sum(float(d) for d in deltas)
                gid = f"legacy{stats['legacy_singletons']}"
                groups[gid] = {"group_id": gid, "clip_id": None, "feature_spec": spec,
                               "candidates": [{"features": feats, "reward": reward,
                                               "reward_source": "legacy", "knobs": {}}],
                               "chosen": 0}
                order.append(gid)
                stats["legacy_singletons"] += 1
    return [groups[g] for g in order], stats


def build_training_arrays(path: str, lambda_dense: float = 1.0,
                          revealed_pref_bonus: float = 1.0):
    np = _numpy()
    groups, stats = load_groups(path)
    prepared = []
    for g in groups:
        spec = g["feature_spec"]
        if spec not in FEATURE_SPECS:
            stats.setdefault("unknown_specs", []).append(spec)
            continue
        X = np.array([to_vector(c["features"], spec) for c in g["candidates"]], dtype=np.float64)
        rewards = []
        for c in g["candidates"]:
            if c.get("overall") is not None and c.get("dense") is not None:
                rewards.append(c["overall"] + lambda_dense * c["dense"])
            else:
                rewards.append(c["reward"])
        r = np.array(rewards, dtype=np.float64)
        # A logged CREATOR (or agent) pick is a revealed preference, not just
        # another critic score. A rubric-derived reward cannot identify per-user
        # taste — the objective has to carry a user-dependent term. The aux
        # preference loss does most of the work here; the bonus additionally makes
        # the revealed pick the group's best reward, so the group-relative
        # advantage agrees with the creator's choice and the dense critic shaping
        # stays interpretable. Ablation (12 neutral briefs, two opposite tastes):
        # no preference term 5/12 differing (direction at chance); preference loss
        # alone 12/12; reward override alone 10/12. See docs/paper-findings.md.
        chosen = g.get("chosen")
        chosen_by = g.get("chosen_by")
        if (chosen is not None and 0 <= chosen < len(r)
                and chosen_by in ("creator", "agent", "human")):
            r = r.copy()
            r[chosen] = float(r.max()) + float(revealed_pref_bonus)
        prepared.append({"group_id": g["group_id"], "clip_id": g["clip_id"], "spec": spec,
                         "X": X, "rewards": r, "A": group_advantage(r),
                         "chosen": chosen, "chosen_by": chosen_by})
    usable = [p for p in prepared if p["X"].shape[0] >= 2]
    stats["groups_total"] = len(prepared)
    stats["groups_usable"] = len(usable)
    stats["groups_singleton"] = len(prepared) - len(usable)
    return prepared, stats


def pairwise_agreement(model: StyleBrain, prepared: list[dict]) -> tuple[float, int]:
    """Fraction of within-group candidate PAIRS the model orders the same way the
    reward does. Chance is 0.5 for any group size, and — unlike argmax accuracy —
    it is not quantised by the group size, so it is the lower-variance statistic
    for judging whether a preference signal was really learned."""
    agree = 0
    total = 0
    for p in prepared:
        r = p["rewards"]
        u = model.utilities(p["X"])
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                if r[i] == r[j]:
                    continue
                total += 1
                if (u[i] - u[j]) * (r[i] - r[j]) > 0:
                    agree += 1
    return ((agree / total) if total else 0.0), total


def ranking_accuracy(model: StyleBrain, prepared: list[dict]) -> tuple[float, int]:
    """Fraction of groups whose best-reward candidate is the model's argmax."""
    if not prepared:
        return 0.0, 0
    hits = 0
    for p in prepared:
        best = int(p["rewards"].argmax())
        pred = int(model.utilities(p["X"]).argmax())
        hits += int(best == pred)
    return hits / len(prepared), len(prepared)


# ---------------------------------------------------------------------------
# training entry point
# ---------------------------------------------------------------------------
DEFAULT_D_H = 24
DEFAULT_D_Z = 16


def train(dataset: str, identity: str, style: str | None = None, creator: str = "default",
          epochs: int = 150, lr_muon: float = 0.005, lr_adamw: float = 0.005,
          temperature: float = 0.5, clip_eps: float = 0.2, entropy_coef: float = 0.01,
          pref_coef: float = 0.5, lambda_dense: float = 1.0, grad_clip: float = 1.0,
          inner_steps: int = 1, batch_groups: int = 16,
          revealed_pref_bonus: float = 1.0,
          d_h: int = DEFAULT_D_H,
          d_z: int = DEFAULT_D_Z, seed: int = 0, holdout_every: int = 5,
          warm_start: bool = True, freeze_style: bool = False,
          verbose: bool = False) -> dict:
    """Train the shared trunk + this creator's latent on the loop's group log.

    Returns metrics; writes `style` (shared) and `identity` (per-creator) GGUFs.
    The one honest guarantee is the reported held-out ranking accuracy: the
    fraction of unseen clips where the model's preferred candidate is the one
    the critic preferred. Publication of the artifact is not a quality claim.

    `freeze_style=True` trains ONLY the per-creator latent against a frozen
    shared trunk and leaves the style-brain artifact untouched — this is the
    architecture the design specifies ("only the identity is per-creator"), and
    it is what lets several creators share one style-brain.

    Defaults are the measured operating point on a synthetic hidden-preference
    benchmark (see tests/test_taste_model.py): Muon's lr is in spectral-norm
    units, so it must be an order of magnitude below a typical Adam lr on tiny
    matrices (lr 0.02 measurably oversteps: training accuracy stalls ~0.5).
    """
    np = _numpy()
    prepared, stats = build_training_arrays(dataset, lambda_dense, revealed_pref_bonus)
    usable = [p for p in prepared if p["X"].shape[0] >= 2]
    if not usable:
        raise SystemExit(
            f"no usable groups in {dataset!r}: {stats}\n"
            "Group-relative training needs at least one clip with >=2 candidates. "
            "Produce them by running the loop with propose_schema group=K "
            "(the plugin logs every group automatically), or legacy flat records "
            "only (which carry no group-relative signal).")

    specs = {p["spec"] for p in usable}
    if len(specs) != 1:
        raise SystemExit(f"mixed feature specs in one dataset: {sorted(specs)}; "
                         "train one modality per style-brain")
    spec = specs.pop()
    d_in = len(FEATURE_SPECS[spec])
    chance = 1.0 / max(2, int(np.median([p["X"].shape[0] for p in usable])))

    # Split by clip when possible so held-out accuracy is per-clip, not
    # per-candidate. The hash MUST be deterministic: python's builtin hash() is
    # salted per process, which would silently make the split (and therefore the
    # reported accuracy) non-reproducible across runs.
    clip_ids = [p["clip_id"] for p in usable]
    if any(c is not None for c in clip_ids) and len({c for c in clip_ids if c is not None}) >= 2:
        eval_idx = {i for i, c in enumerate(clip_ids)
                    if c is not None
                    and int(hashlib.sha1(str(c).encode()).hexdigest()[:8], 16) % holdout_every == 0}
    else:
        eval_idx = {i for i in range(len(usable)) if (i % holdout_every) == 0}
    train_set = [p for i, p in enumerate(usable) if i not in eval_idx]
    eval_set = [p for i, p in enumerate(usable) if i in eval_idx]
    if not train_set:
        train_set, eval_set = usable, []

    model = StyleBrain(d_in, d_h, d_z, seed=seed)
    style_path = style or os.path.join(os.path.dirname(os.path.abspath(identity)),
                                       f"style_brain_{spec.replace('/', '_')}.gguf")
    warm = False
    if warm_start and os.path.exists(style_path):
        model.load_shared(style_path)
        warm = True
    if warm_start and os.path.exists(identity):
        try:
            z = read_gguf(identity)["tensors"].get("z_u")
            if z is not None and z.shape == model.p["z_u"].shape:
                model.p["z_u"] = np.asarray(z, dtype=np.float64)
        except (ValueError, KeyError):
            pass

    if freeze_style and not os.path.exists(style_path):
        raise SystemExit(
            f"--freeze-style needs an existing shared style-brain at {style_path!r}; "
            "train the shared trunk first (a normal run), then train identities")
    model.muon.lr, model.adamw.lr = lr_muon, lr_adamw
    acc0, n_eval = ranking_accuracy(model, eval_set)
    pair0, n_pairs = pairwise_agreement(model, eval_set)
    history = []
    rng = np.random.default_rng(seed)
    bs = max(1, int(batch_groups))
    for epoch in range(max(1, epochs)):
        n_batches = max(1, (len(train_set) + bs - 1) // bs)
        order = rng.permutation(n_batches)
        step_metrics = {"loss": 0.0, "groups": 0, "clip_fraction": 0.0,
                        "mean_abs_rho_minus_1": 0.0, "grad_norm": 0.0, "clip_hits": 0.0}
        n_steps = 0
        for b in order:
            chunk = train_set[b * bs:(b + 1) * bs]
            # GRPO baseline: the ratio reference is frozen for THIS minibatch, not
            # for the whole epoch. A per-epoch baseline is what makes
            # rho = exp(cumulative logit drift) explode; with the default single
            # inner step the ratio is 1 by construction, so the observed
            # clip_fraction is ~0 (honest: the clip is a safety net for
            # inner_steps > 1, it is not what makes this objective converge).
            olds = [policy_logits(model, p["X"], temperature)["s"] for p in chunk]
            acc_grads: dict = {}
            for p, old_i in zip(chunk, olds):
                for _inner in range(max(1, inner_steps)):
                    g = {"X": p["X"], "A": p["A"], "chosen": p.get("chosen")}
                    grads: dict = {}
                    m = grpo_loss_and_grads(model, [g], [old_i], temperature=temperature,
                                            clip_eps=clip_eps, entropy_coef=entropy_coef,
                                            pref_coef=pref_coef, model_grads_out=grads)
                    for k, v in grads.items():
                        acc_grads[k] = v if k not in acc_grads else acc_grads[k] + v
                    for k in step_metrics:
                        if k in m:
                            step_metrics[k] += m[k]
            for k in acc_grads:
                acc_grads[k] = acc_grads[k] / len(chunk)
            if freeze_style:
                # Only the identity latent is per-creator; the shared trunk is
                # frozen, so drop every gradient except z_u.
                acc_grads = {"z_u": acc_grads["z_u"]}
            model.zero_grad()
            model.add_grads(acc_grads)
            applied = model.apply_grads(grad_clip)
            step_metrics["grad_norm"] += applied["grad_norm"]
            step_metrics["clip_hits"] += 1.0 if applied["clipped"] else 0.0
            n_steps += 1
            if verbose:
                print(f"  epoch {epoch} batch {b}: loss "
                      f"{step_metrics['loss'] / max(1, len(chunk)):+.4f} "
                      f"grad_norm {applied['grad_norm']:.4f}", file=sys.stderr)
        if n_steps:
            for k in step_metrics:
                step_metrics[k] /= n_steps
        step_metrics["epoch"] = epoch
        step_metrics["train_rank_acc"] = ranking_accuracy(model, train_set)[0]
        step_metrics["eval_rank_acc"] = ranking_accuracy(model, eval_set)[0]
        step_metrics["eval_pair_agree"] = pairwise_agreement(model, eval_set)[0]
        step_metrics["train_pair_agree"] = pairwise_agreement(model, train_set)[0]
        step_metrics["z_u_norm"] = float(np.linalg.norm(model.p["z_u"]))
        history.append(step_metrics)

    acc1, n_eval = ranking_accuracy(model, eval_set)
    pair1, _ = pairwise_agreement(model, eval_set)
    if freeze_style and os.path.exists(style_path):
        # Do NOT rewrite the shared trunk: other creators depend on it.
        shared = {"path": style_path, "bytes": os.path.getsize(style_path),
                  "tensors": [], "metadata_keys": [], "frozen": True}
    else:
        shared = write_gguf(style_path, {
            "general.architecture": "editapart-taste",
            "general.name": f"style-brain/{spec}",
            "editapart.kind": "style_brain",
            "editapart.feature_spec": spec,
            "editapart.features": FEATURE_SPECS[spec],
            "editapart.d_in": d_in, "editapart.d_h": d_h, "editapart.d_z": d_z,
            "editapart.optimizer_2d": "muon-newton-schulz-5",
            "editapart.optimizer_1d": "adamw",
            "editapart.model_layout": MODEL_LAYOUT,
            "editapart.trained_at": int(time.time()),
        }, model.shared_tensors())
    digest = model.shared_digest()
    ident = write_gguf(identity, {
        "general.architecture": "editapart-taste",
        "general.name": creator,
        "editapart.kind": "identity",
        "editapart.creator": creator,
        "editapart.feature_spec": spec,
        "editapart.d_in": d_in, "editapart.d_h": d_h, "editapart.d_z": d_z,
        "editapart.style_file": os.path.basename(style_path),
        "editapart.style_digest": digest,
        "editapart.model_layout": MODEL_LAYOUT,
        "editapart.style_frozen": bool(freeze_style),
        "editapart.revealed_pref_bonus": float(revealed_pref_bonus),
        "editapart.trained_groups": len(train_set),
        "editapart.eval_groups": len(eval_set),
        "editapart.eval_rank_acc": float(acc1),
        "editapart.eval_pair_agree": float(pair1),
        "editapart.trained_at": int(time.time()),
    }, {"z_u": model.p["z_u"]})

    return {
        "ok": True,
        "dataset": dataset,
        "identity": identity,
        "style_brain": style_path,
        "creator": creator,
        "feature_spec": spec,
        "dims": {"d_in": d_in, "d_h": d_h, "d_z": d_z},
        "warm_start": warm,
        "style_frozen": bool(freeze_style),
        "revealed_pref_bonus": float(revealed_pref_bonus),
        "groups": stats,
        "train_groups": len(train_set),
        "eval_groups": len(eval_set),
        "eval_rank_acc_before": round(float(acc0), 4),
        "eval_rank_acc_after": round(float(acc1), 4),
        "eval_pair_agree_before": round(float(pair0), 4),
        "eval_pair_agree_after": round(float(pair1), 4),
        "eval_pairs": n_pairs,
        "chance_pair_agree": 0.5,
        "chance_rank_acc": round(chance, 4),
        "z_u_norm": round(float(np.linalg.norm(model.p["z_u"])), 6),
        "z_u": [round(float(v), 6) for v in model.p["z_u"]],
        "style_digest": digest,
        "style_bytes": shared["bytes"],
        "identity_bytes": ident["bytes"],
        "history": history,
        "artifacts": {"style_brain": shared, "identity": ident},
    }


# ---------------------------------------------------------------------------
# scoring (inference) — used by the loop to pick among candidate schemas
# ---------------------------------------------------------------------------
class TasteScorer:
    """Load a per-creator identity (+ its shared style-brain) and score edits."""

    def __init__(self, model: StyleBrain, meta: dict, style_meta: dict, identity_path: str):
        self.model = model
        self.meta = meta
        self.style_meta = style_meta
        self.identity_path = identity_path

    @classmethod
    def load(cls, identity: str, style: str | None = None) -> "TasteScorer":
        art = read_gguf(identity)
        meta = art["metadata"]
        if meta.get("editapart.kind") != "identity":
            raise ValueError(f"{identity}: not an identity artifact")
        layout = meta.get("editapart.model_layout")
        if layout != MODEL_LAYOUT:
            raise ValueError(
                f"{identity}: identity model layout {layout!r} does not match this build "
                f"({MODEL_LAYOUT}); train a new identity (train_identity) to upgrade it")
        if "z_u" not in art["tensors"]:
            raise ValueError(f"{identity}: missing z_u tensor")
        d_in = int(meta["editapart.d_in"])
        d_h = int(meta["editapart.d_h"])
        d_z = int(meta["editapart.d_z"])
        style_path = style or os.environ.get("DSH_EDITAPART_STYLE") or os.path.join(
            os.path.dirname(os.path.abspath(identity)), str(meta.get("editapart.style_file", "")))
        if not os.path.exists(style_path):
            raise FileNotFoundError(
                f"style-brain {style_path!r} referenced by {identity!r} is missing; "
                f"train it (taste_model.py train) or pass --style/DSH_EDITAPART_STYLE")
        model = StyleBrain(d_in, d_h, d_z)
        style_meta = model.load_shared(style_path)
        digest = model.shared_digest()
        recorded = meta.get("editapart.style_digest")
        if recorded and recorded != digest:
            raise ValueError(
                f"identity {identity!r} was trained against style-brain digest {recorded[:12]}… "
                f"but {style_path!r} has {digest[:12]}…; retrain against the matching trunk")
        model.p["z_u"] = _numpy().asarray(art["tensors"]["z_u"], dtype=_numpy().float64).reshape(-1)
        return cls(model, meta, style_meta, identity)

    @property
    def feature_spec(self) -> str:
        return str(self.meta.get("editapart.feature_spec", FEATURE_SPEC_VIDEO))

    def score_features(self, feats: dict) -> float:
        x = _numpy().array([to_vector(feats, self.feature_spec)], dtype=_numpy().float64)
        return float(self.model.utilities(x)[0])

    def score_many(self, feats_list: list[dict]) -> list[float]:
        np = _numpy()
        X = np.array([to_vector(f, self.feature_spec) for f in feats_list], dtype=np.float64)
        return [float(v) for v in self.model.utilities(X)]

    def info(self) -> dict:
        np = _numpy()
        return {
            "identity": self.identity_path,
            "creator": self.meta.get("editapart.creator"),
            "feature_spec": self.feature_spec,
            "style_file": self.meta.get("editapart.style_file"),
            "style_digest": self.meta.get("editapart.style_digest"),
            "dims": {"d_in": self.model.d_in, "d_h": self.model.d_h, "d_z": self.model.d_z},
            "trained_groups": self.meta.get("editapart.trained_groups"),
            "eval_groups": self.meta.get("editapart.eval_groups"),
            "eval_rank_acc": self.meta.get("editapart.eval_rank_acc"),
            "z_u_norm": round(float(np.linalg.norm(self.model.p["z_u"])), 6),
            "z_u": [round(float(v), 6) for v in self.model.p["z_u"]],
            "features": FEATURE_SPECS[self.feature_spec],
        }


def init_identity(identity: str, style: str | None = None, creator: str = "default",
                  d_h: int = DEFAULT_D_H, d_z: int = DEFAULT_D_Z,
                  spec: str = FEATURE_SPEC_VIDEO, seed: int = 0) -> dict:
    """Create a zero-latent identity bound to a (possibly fresh) style-brain, so
    a brand-new creator can run the loop immediately and train from zero."""
    d_in = len(FEATURE_SPECS[spec])
    style_path = style or os.path.join(os.path.dirname(os.path.abspath(identity)),
                                       f"style_brain_{spec.replace('/', '_')}.gguf")
    if os.path.exists(style_path):
        model = StyleBrain(d_in, d_h, d_z, seed=seed)
        model.load_shared(style_path)
    else:
        model = StyleBrain(d_in, d_h, d_z, seed=seed)
        write_gguf(style_path, {
            "general.architecture": "editapart-taste",
            "general.name": f"style-brain/{spec}",
            "editapart.kind": "style_brain",
            "editapart.feature_spec": spec,
            "editapart.features": FEATURE_SPECS[spec],
            "editapart.d_in": d_in, "editapart.d_h": d_h, "editapart.d_z": d_z,
            "editapart.optimizer_2d": "muon-newton-schulz-5",
            "editapart.optimizer_1d": "adamw",
            "editapart.model_layout": MODEL_LAYOUT,
            "editapart.initialized_at": int(time.time()),
        }, model.shared_tensors())
    digest = model.shared_digest()
    out = write_gguf(identity, {
        "general.architecture": "editapart-taste",
        "general.name": creator,
        "editapart.kind": "identity",
        "editapart.creator": creator,
        "editapart.feature_spec": spec,
        "editapart.d_in": d_in, "editapart.d_h": d_h, "editapart.d_z": d_z,
        "editapart.style_file": os.path.basename(style_path),
        "editapart.style_digest": digest,
        "editapart.model_layout": MODEL_LAYOUT,
        "editapart.style_frozen": False,
        "editapart.revealed_pref_bonus": 1.0,
        "editapart.trained_groups": 0,
        "editapart.initialized_at": int(time.time()),
    }, {"z_u": model.p["z_u"]})
    return {"ok": True, "identity": identity, "style_brain": style_path,
            "creator": creator, "feature_spec": spec, "z_u": [0.0] * d_z,
            "fresh": True, "identity_bytes": out["bytes"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_json_arg(value: str) -> dict:
    if os.path.exists(value):
        with open(value, encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="taste_model.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train the shared trunk + this creator's z_u")
    p.add_argument("--dataset", required=True)
    p.add_argument("--identity", required=True)
    p.add_argument("--style", default=None)
    p.add_argument("--creator", default="default")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr-muon", type=float, default=0.005)
    p.add_argument("--lr-adamw", type=float, default=0.005)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--pref-coef", type=float, default=0.5)
    p.add_argument("--lambda-dense", type=float, default=1.0)
    p.add_argument("--d-h", type=int, default=DEFAULT_D_H)
    p.add_argument("--d-z", type=int, default=DEFAULT_D_Z)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--batch-groups", type=int, default=16)
    p.add_argument("--revealed-pref-bonus", type=float, default=1.0)
    p.add_argument("--holdout-every", type=int, default=5)
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--freeze-style", action="store_true",
                   help="train only z_u against a frozen shared style-brain")
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("score", help="score candidate features with an identity")
    p.add_argument("--identity", required=True)
    p.add_argument("--style", default=None)
    p.add_argument("--features", default=None, help="JSON dict (or JSON list of dicts)")
    p.add_argument("--schema", default=None)
    p.add_argument("--inventory", default=None)
    p.add_argument("--rubric", default=None)

    p = sub.add_parser("info", help="identity metadata + latent")
    p.add_argument("--identity", required=True)
    p.add_argument("--style", default=None)

    p = sub.add_parser("init", help="create a zero-latent identity")
    p.add_argument("--identity", required=True)
    p.add_argument("--style", default=None)
    p.add_argument("--creator", default="default")
    p.add_argument("--spec", default=FEATURE_SPEC_VIDEO)

    p = sub.add_parser("features", help="compute the feature vector for a schema")
    p.add_argument("--schema", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--rubric", required=True)
    p.add_argument("--spec", default=FEATURE_SPEC_VIDEO)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "train":
            out = train(args.dataset, args.identity, style=args.style, creator=args.creator,
                        epochs=args.epochs, lr_muon=args.lr_muon, lr_adamw=args.lr_adamw,
                        temperature=args.temperature, clip_eps=args.clip_eps,
                        entropy_coef=args.entropy_coef, pref_coef=args.pref_coef,
                        lambda_dense=args.lambda_dense, grad_clip=args.grad_clip,
                        inner_steps=args.inner_steps, batch_groups=args.batch_groups,
                        revealed_pref_bonus=args.revealed_pref_bonus,
                        d_h=args.d_h, d_z=args.d_z,
                        seed=args.seed, holdout_every=args.holdout_every,
                        warm_start=not args.no_warm_start,
                        freeze_style=args.freeze_style, verbose=args.verbose)
        elif args.cmd == "score":
            sc = TasteScorer.load(args.identity, args.style)
            if args.features:
                payload = json.loads(args.features)
                if isinstance(payload, list):
                    out = {"scores": sc.score_many(payload), "feature_spec": sc.feature_spec}
                else:
                    out = {"score": sc.score_features(payload), "feature_spec": sc.feature_spec}
            else:
                if not (args.schema and args.inventory and args.rubric):
                    ap.error("score needs --features or --schema/--inventory/--rubric")
                feats = features_for(sc.feature_spec, _load_json_arg(args.schema),
                                     _load_json_arg(args.inventory), _load_json_arg(args.rubric))
                out = {"score": sc.score_features(feats), "features": feats,
                       "feature_spec": sc.feature_spec}
        elif args.cmd == "info":
            out = TasteScorer.load(args.identity, args.style).info()
        elif args.cmd == "init":
            out = init_identity(args.identity, style=args.style, creator=args.creator,
                                spec=args.spec)
        elif args.cmd == "features":
            feats = features_for(args.spec, _load_json_arg(args.schema),
                                 _load_json_arg(args.inventory), _load_json_arg(args.rubric))
            out = {"feature_spec": args.spec, "features": feats,
                   "vector": to_vector(feats, args.spec)}
        else:
            ap.error(f"unknown command {args.cmd}")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced to the engine as a tool error
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
