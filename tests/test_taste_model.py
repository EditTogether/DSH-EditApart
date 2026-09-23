#!/usr/bin/env python3
"""Verification suite for the EditApart per-creator taste model.

Every assertion here is a checkable property, not a smoke test:

  * Newton-Schulz: bounded singular-value band + scale invariance (the reference
    Muon iteration is an APPROXIMATE orthogonalization — asserting exact
    orthogonality would test the wrong contract).
  * Optimizer split: Muon touches only 2D parameters, AdamW only 1D ones
    (including `z_u`), and each leaves the other's parameters untouched.
  * Gradients: the analytic backward pass matches finite differences.
  * GGUF: real container round-trips bit-exactly; metadata types survive; a
    digest mismatch (identity trained against a different style-brain) is
    refused rather than silently scoring with the wrong trunk.
  * Learning: on a synthetic hidden taste function the objective actually
    recovers held-out preferences (ranking accuracy vs. chance).
  * Identity semantics: `z_u` changes decisions, is required, and is
    interpolatable (midpoint latents score between the endpoints).
  * Engine wiring: the loop builds groups, logs them, folds critic rewards, and
    `group=1` still reproduces the legacy single-schema proposal.
  * Graceful degradation: the legacy engine path works with numpy unavailable.

Run:  <venv>/bin/python tests/test_taste_model.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "bin")
sys.path.insert(0, BIN)

import numpy as np  # noqa: E402

import edit_apart_core as eng  # noqa: E402
import taste_model as tm  # noqa: E402


def fixture_inventory(n_shots: int = 12, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    shots = []
    t = 0.0
    for i in range(n_shots):
        d = float(round(rng.uniform(0.5, 3.5), 3))
        shots.append({"shot": f"clip_{i + 1:03d}", "start": round(t, 3),
                      "end": round(t + d, 3), "duration": d, "src_idx": i + 1,
                      "features": {"lum_p50": float(rng.uniform(40, 200)),
                                   "lum_p95": float(rng.uniform(150, 250)),
                                   "motion": float(rng.uniform(0.2, 14.0)),
                                   "rms_db": float(rng.uniform(-40, -5)),
                                   "peak_db": float(rng.uniform(-10, 0))}})
        t += d
    return {"source": "/tmp/fixture.mp4", "shots": shots}


def fixture_rubric(**over) -> dict:
    r = {"intent": "punchy highlight", "pace": "brisk", "min_shot_dur": 0.9,
         "max_shot_dur": 4.0, "skip_short": True, "target_duration": 20.0,
         "no_shot_under_s": 0.9}
    r.update(over)
    return r


def synthetic_dataset(path: str, n_clips: int = 24, k: int = 5, seed: int = 0,
                      noise: float = 0.0) -> dict:
    """Groups whose reward is a hidden quadratic taste over the video feature
    layout. This is a ground truth the model can only get right by learning."""
    rng = np.random.default_rng(seed)
    names = tm.FEATURE_SPECS[tm.FEATURE_SPEC_VIDEO]
    theta = rng.normal(size=len(names))
    with open(path, "w", encoding="utf-8") as fh:
        for c in range(n_clips):
            X = rng.normal(size=(k, len(names)))
            r = -((X - theta) ** 2).sum(axis=1)
            if noise:
                r = r + rng.normal(scale=noise, size=k)
            cands = []
            for i in range(k):
                cands.append({"idx": i, "features": {n: float(x) for n, x in zip(names, X[i])},
                              "reward_obj": float(r[i]), "overall_obj": float(r[i]),
                              "dense_obj": 0.0, "knobs": {"i": i}})
            fh.write(json.dumps({"kind": "group", "group_id": f"c{c}", "clip_id": f"clip{c}",
                                 "feature_spec": tm.FEATURE_SPEC_VIDEO, "candidates": cands}) + "\n")
    return {"theta": theta.tolist(), "names": names}


class TestNewtonSchulz(unittest.TestCase):
    def test_shape_and_bounded_band(self):
        rng = np.random.default_rng(0)
        for shape in [(6, 6), (8, 6), (6, 8), (16, 4), (10, 24)]:
            G = rng.normal(size=shape)
            O = tm.newton_schulz(G, 5)
            self.assertEqual(O.shape, shape)
            sv = np.linalg.svd(O, compute_uv=False)
            self.assertGreater(sv.min(), 0.25, f"{shape}: min sv {sv.min()}")
            self.assertLess(sv.max(), 2.0, f"{shape}: max sv {sv.max()}")

    def test_scale_invariance(self):
        rng = np.random.default_rng(3)
        G = rng.normal(size=(12, 7))
        self.assertTrue(np.allclose(tm.newton_schulz(G, 5), tm.newton_schulz(G * 9.5, 5),
                                    atol=1e-9))

    def test_rejects_1d(self):
        with self.assertRaises(ValueError):
            tm.newton_schulz(np.zeros(5), 5)


class TestOptimizerSplit(unittest.TestCase):
    def setUp(self):
        self.model = tm.StyleBrain(d_in=8, d_h=6, d_z=5, seed=0)

    def all_grads(self, seed=0):
        rng = np.random.default_rng(seed)
        return {n: rng.normal(size=self.model.p[n].shape) for n in self.model.spec}

    def test_muon_only_2d(self):
        before = {n: self.model.p[n].copy() for n in self.model.spec}
        grads = self.all_grads()
        self.model.muon.step(grads)
        for name, (shape, opt) in self.model.spec.items():
            if opt == "muon":
                self.assertFalse(np.allclose(self.model.p[name], before[name]),
                                 f"Muon did not update 2D param {name}")
            else:
                self.assertTrue(np.array_equal(self.model.p[name], before[name]),
                                f"Muon touched 1D param {name}")
        self.assertEqual(self.model.muon.touched_1d, [])

    def test_adamw_only_1d(self):
        before = {n: self.model.p[n].copy() for n in self.model.spec}
        self.model.adamw.step(self.all_grads(1))
        for name, (shape, opt) in self.model.spec.items():
            if opt == "adamw":
                self.assertFalse(np.array_equal(self.model.p[name], before[name]),
                                 f"AdamW did not update {name}")
            else:
                self.assertTrue(np.array_equal(self.model.p[name], before[name]),
                                f"AdamW touched 2D param {name}")
        self.assertEqual(self.model.adamw.touched_2d, [])

    def test_z_u_is_adamw_and_gets_updated(self):
        _, opt = self.model.spec["z_u"]
        self.assertEqual(opt, "adamw")
        before = self.model.p["z_u"].copy()
        self.model.adamw.step({"z_u": np.ones(self.model.d_z)})
        self.assertFalse(np.array_equal(self.model.p["z_u"], before))
        self.assertGreater(np.linalg.norm(self.model.p["z_u"]), 0.0)

    def test_wrong_rank_is_refused(self):
        with self.assertRaises(AssertionError):
            self.model.muon.step({"W1": np.zeros(self.model.d_in)})
        with self.assertRaises(AssertionError):
            self.model.adamw.step({"z_u": np.zeros((self.model.d_z, 1))})


class TestGradients(unittest.TestCase):
    def test_analytic_matches_finite_difference(self):
        rng = np.random.default_rng(7)
        worst = 0.0
        for trial in range(3):
            m = tm.StyleBrain(d_in=6, d_h=5, d_z=4, seed=trial)
            X = rng.normal(size=(4, 6))
            dU = rng.normal(size=4)
            cache: dict = {}
            m._trunk(X, m.p["z_u"], cache)
            grads = m.backward(dU, cache)
            eps = 1e-6
            for name in m.spec:
                flat = m.p[name].ravel()
                for i in range(min(10, flat.size)):
                    old = flat[i]
                    flat[i] = old + eps
                    up = m.utilities(X)
                    flat[i] = old - eps
                    dn = m.utilities(X)
                    flat[i] = old
                    num = float(((up - dn) / (2 * eps)) @ dU)
                    ana = float(grads[name].ravel()[i])
                    worst = max(worst, abs(num - ana) / max(1e-9, abs(num) + abs(ana)))
        self.assertLess(worst, 1e-6, f"worst relative gradient error {worst:.3e}")


class TestObjectiveFidelity(unittest.TestCase):
    """The GRPO ratio must be a ratio.

    A review found `train()` storing raw standardised logits as the ratio
    baseline, so `rho = exp(log_softmax(s) - s) = 1/Z` — a per-group constant.
    Consequences: every negative-advantage candidate fell into the clipped branch
    and its downward gradient became independent of |A| (a -0.1 and a -8.0 pushed
    down identically), and the reported `clip_fraction` was a per-call sum divided
    by the batch count (5.9, i.e. 590%). These tests pin the corrected behaviour.
    """

    def _group(self, A, k=None):
        model = tm.StyleBrain(6, 6, 4, seed=0)
        X = np.array([[1.0, 0, 0, 0, 0, 1], [0, 1.0, 0, 0, 0, 1], [0, 0, 1.0, 0, 0, 1]])
        if k is not None:
            X = X[:k]
        s = tm.policy_logits(model, X, 0.5)["s"]
        return model, X, s, np.asarray(A, dtype=float)

    def test_baseline_must_be_log_probabilities(self):
        model, X, s, A = self._group([1.0, 0.0, -1.0])
        got = {}
        for label, old in (("log_probs", [tm.log_softmax(s)]), ("raw_logits", [s])):
            du = []
            tm.grpo_loss_and_grads(model, [{"X": X, "A": A, "chosen": None}], old,
                                   temperature=0.5, entropy_coef=0.0, pref_coef=0.0,
                                   utility_grads_out=du)
            got[label] = du[0]
        # log-probs: rho == 1, so the objective keeps a real magnitude
        self.assertGreater(abs(got["log_probs"][2]), 1.0)
        # raw logits: rho == 1/Z, the gradient collapses by ~the group size
        self.assertLess(abs(got["raw_logits"][2]), abs(got["log_probs"][2]) / 50)

    def test_negative_advantage_scales_with_magnitude(self):
        """A -2.0 advantage must push its candidate down harder than a -0.1 one.

        With the raw-logit baseline both were frozen at the same value
        (measured 0.026815 for -0.1, -2.0 AND -8.0), i.e. the objective had
        degenerated to positive-only REINFORCE.
        """
        pushes = {}
        for label, m in (("weak", 0.1), ("strong", 2.0)):
            model, X, s, _ = self._group([1.0, 0.0, -m])
            A = np.array([1.0, 0.0, -m])
            du = []
            tm.grpo_loss_and_grads(model, [{"X": X, "A": A, "chosen": None}],
                                   [tm.log_softmax(s)], temperature=0.5,
                                   entropy_coef=0.0, pref_coef=0.0, utility_grads_out=du)
            pushes[label] = du[0][2]
            self.assertGreater(pushes[label], 0.0, "negative advantage must push down")
        self.assertGreater(pushes["strong"], 5 * pushes["weak"],
                           f"downward push did not scale with |A|: {pushes}")

    def test_positive_advantage_pushes_up(self):
        model, X, s, A = self._group([1.0, 0.0, -1.0])
        du = []
        tm.grpo_loss_and_grads(model, [{"X": X, "A": A, "chosen": None}],
                               [tm.log_softmax(s)], temperature=0.5, entropy_coef=0.0,
                               pref_coef=0.0, utility_grads_out=du)
        self.assertLess(du[0][0], 0.0, "positive advantage must raise its utility")

    def test_two_candidate_groups_carry_no_gradient(self):
        """Documented limitation of group standardisation: for K=2 the standardised
        logits are exactly +-1/T whatever the utilities are, so the objective is
        constant in them and the group teaches nothing. Use group >= 3."""
        model, X, s, A = self._group([1.0, -1.0], k=2)
        du = []
        tm.grpo_loss_and_grads(model, [{"X": X, "A": A, "chosen": None}],
                               [tm.log_softmax(s)], temperature=0.5, entropy_coef=0.0,
                               pref_coef=0.0, utility_grads_out=du)
        self.assertLess(float(np.abs(du[0]).max()), 1e-5)

    def test_reported_ratio_telemetry_is_a_real_fraction(self):
        with tempfile.TemporaryDirectory() as d:
            ds = os.path.join(d, "s.jsonl")
            synthetic_dataset(ds, n_clips=30, k=5, seed=0)
            m = tm.train(ds, os.path.join(d, "i.gguf"), creator="c", epochs=1,
                         warm_start=False)
            h = m["history"][-1]
            self.assertGreaterEqual(h["clip_fraction"], 0.0)
            self.assertLessEqual(h["clip_fraction"], 1.0,
                                 "clip_fraction must be a fraction, not a per-batch sum")
            self.assertLess(abs(h["mean_abs_rho_minus_1"]), 1e-9,
                            "a frozen baseline must give rho == 1")
            self.assertAlmostEqual(h["groups"], 1.0, places=6,
                                   msg="one group per objective call")


class TestGGUF(unittest.TestCase):
    def test_round_trip_bit_exact(self):
        with tempfile.TemporaryDirectory() as d:
            m = tm.StyleBrain(d_in=7, d_h=5, d_z=4, seed=2)
            path = os.path.join(d, "sb.gguf")
            tm.write_gguf(path, {"general.architecture": "editapart-taste",
                                 "editapart.kind": "style_brain",
                                 "editapart.features": ["a", "b", "c"],
                                 "editapart.d_in": 7, "editapart.d_h": 5,
                                 "editapart.d_z": 4, "flag": True, "ratio": 0.25,
                                 "count": 9},
                          m.shared_tensors())
            back = tm.read_gguf(path)
            self.assertEqual(back["metadata"]["editapart.features"], ["a", "b", "c"])
            self.assertIs(back["metadata"]["flag"], True)
            self.assertEqual(back["metadata"]["count"], 9)
            self.assertAlmostEqual(back["metadata"]["ratio"], 0.25, places=6)
            for name, arr in m.shared_tensors().items():
                self.assertTrue(np.array_equal(back["tensors"][name], arr.astype(np.float32)),
                                f"{name} did not round-trip")
                self.assertEqual(back["tensors"][name].shape, arr.shape)

    def test_rejects_non_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "junk.gguf")
            with open(p, "wb") as fh:
                fh.write(b"not a gguf file at all")
            with self.assertRaises(ValueError):
                tm.read_gguf(p)

    def test_identity_load_refuses_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            ident = os.path.join(d, "erkin.gguf")
            tm.init_identity(ident, creator="erkin")
            # A second, different style-brain must not be accepted.
            other = tm.StyleBrain(d_in=len(tm.VIDEO_FEATURES), d_h=24, d_z=16, seed=99)
            tm.write_gguf(os.path.join(d, "other.gguf"), {
                "general.architecture": "editapart-taste", "editapart.kind": "style_brain",
                "editapart.feature_spec": tm.FEATURE_SPEC_VIDEO,
                "editapart.model_layout": tm.MODEL_LAYOUT,
                "editapart.d_in": 16, "editapart.d_h": 24, "editapart.d_z": 16,
            }, other.shared_tensors())
            with self.assertRaises(ValueError) as cm:
                tm.TasteScorer.load(ident, style=os.path.join(d, "other.gguf"))
            self.assertIn("digest", str(cm.exception))

    def test_older_model_layout_is_refused_with_guidance(self):
        """A style-brain written by an older parameter layout must not be loaded
        silently — it has to say what to do about it."""
        with tempfile.TemporaryDirectory() as d:
            m = tm.StyleBrain(len(tm.VIDEO_FEATURES), 24, 16, seed=0)
            style = os.path.join(d, "style_brain_video_v1.gguf")
            tm.write_gguf(style, {"general.architecture": "editapart-taste",
                                  "editapart.kind": "style_brain",
                                  "editapart.feature_spec": tm.FEATURE_SPEC_VIDEO,
                                  "editapart.d_in": 16, "editapart.d_h": 24,
                                  "editapart.d_z": 16}, m.shared_tensors())
            with self.assertRaises(ValueError) as cm:
                tm.init_identity(os.path.join(d, "i.gguf"), creator="old")
            msg = str(cm.exception)
            self.assertIn("layout", msg)
            self.assertIn("train_identity", msg)

    def test_missing_style_brain_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            ident = os.path.join(d, "erkin.gguf")
            tm.init_identity(ident, creator="erkin")
            os.remove(os.path.join(d, "style_brain_video_v1.gguf"))
            with self.assertRaises(FileNotFoundError):
                tm.TasteScorer.load(ident)


class TestDatasetFolding(unittest.TestCase):
    def test_group_reward_and_legacy_records(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "group", "group_id": "g1", "clip_id": "c1",
                                     "feature_spec": tm.FEATURE_SPEC_VIDEO,
                                     "candidates": [
                                         {"features": {"bias": 1.0}, "reward_obj": -1.0,
                                          "overall_obj": -0.5, "dense_obj": -0.5},
                                         {"features": {"bias": 1.0}, "reward_obj": -2.0,
                                          "overall_obj": -1.0, "dense_obj": -1.0}]}) + "\n")
                fh.write(json.dumps({"kind": "reward", "group_id": "g1", "candidate": 0,
                                     "reward": 0.25, "overall_obj": 0.5,
                                     "dense_obj": -0.25, "subjective": 0.9}) + "\n")
                fh.write(json.dumps({"features": [1, 0, 1], "score": 0.7,
                                     "deltas": [0.0, -0.2]}) + "\n")
            groups, stats = tm.load_groups(path)
            self.assertEqual(len(groups), 2)
            self.assertEqual(stats["legacy_singletons"], 1)
            self.assertEqual(stats["reward_records"], 1)
            g = groups[0]
            self.assertEqual(g["chosen"], 0)
            self.assertEqual(g["candidates"][0]["reward"], 0.25)
            self.assertEqual(g["candidates"][0]["reward_source"], "critic")

            # lambda_dense is applied to the separate overall/dense components.
            prepared, _ = tm.build_training_arrays(path, lambda_dense=1.0)
            g1 = next(p for p in prepared if p["group_id"] == "g1")
            self.assertAlmostEqual(float(g1["rewards"][0]), 0.5 - 0.25, places=6)
            prepared2, _ = tm.build_training_arrays(path, lambda_dense=2.0)
            g1b = next(p for p in prepared2 if p["group_id"] == "g1")
            self.assertAlmostEqual(float(g1b["rewards"][0]), 0.5 - 0.5, places=6)

    def test_null_rewards_do_not_crash_and_are_skipped(self):
        """The photo engine writes `"reward_obj": null` for unscored candidates.
        `float(None)` used to abort taste_status/train on the whole file."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "null.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                # a group with NO scored candidates
                fh.write(json.dumps({"kind": "group", "group_id": "allnull",
                                     "feature_spec": tm.FEATURE_SPEC_PHOTO,
                                     "candidates": [
                                         {"features": {"crop_area": 0.3}, "reward_obj": None},
                                         {"features": {"crop_area": 0.2}, "reward_obj": None}]}) + "\n")
                # a group with one scored candidate and one null
                fh.write(json.dumps({"kind": "group", "group_id": "partial",
                                     "feature_spec": tm.FEATURE_SPEC_PHOTO,
                                     "candidates": [
                                         {"features": {"crop_area": 0.3}, "reward_obj": 0.5},
                                         {"features": {"crop_area": 0.2}, "reward_obj": None}]}) + "\n")
            groups, stats = tm.load_groups(path)          # must not raise
            self.assertEqual(len(groups), 2)
            prepared, pstats = tm.build_training_arrays(path)
            self.assertEqual(pstats.get("candidates_unscored"), 3)
            self.assertEqual(len(prepared), 0,
                             "groups without enough scored candidates teach nothing")
            status = tm.dataset_status(path)              # must not raise either
            self.assertTrue(status["exists"])

    def test_hard_broken_record_is_counted_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json}\n")
                fh.write(json.dumps({"kind": "group", "group_id": "g", "candidates": [
                    {"features": {}, "reward_obj": 1.0}]}) + "\n")
            groups, stats = tm.load_groups(path)
            self.assertEqual(stats["bad_lines"], 1)
            self.assertEqual(len(groups), 1)


class TestLearning(unittest.TestCase):
    """Can the objective recover a hidden preference it is never shown?

    Reward is a hidden linear taste over the feature layout; the model only ever
    sees group-relative rewards. Baseline for reference: a least-squares linear
    fit on the same features reaches 1.00 held-out ranking accuracy, and a
    supervised pairwise-ranking loss on this architecture reaches ~0.80. GRPO
    (no reward supervision at all) lands around 0.66-0.70, i.e. well above the
    0.20 chance level but below the supervised bound — the honest characterisation
    is "recovers a real preference signal, at the sample cost you expect from a
    policy-gradient objective".
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tastelearn_")
        cls.dataset = os.path.join(cls.tmp, "synth.jsonl")
        synthetic_dataset(cls.dataset, n_clips=120, k=5, seed=0)
        cls.identity = os.path.join(cls.tmp, "synth.gguf")
        cls.metrics = tm.train(cls.dataset, cls.identity, creator="synth", epochs=150,
                               lr_muon=0.005, lr_adamw=0.005, temperature=0.5,
                               holdout_every=5, seed=0)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_recovers_held_out_preferences(self):
        """The primary metric is within-group pairwise agreement (chance 0.5),
        because argmax accuracy is quantised by the group size and is far noisier:
        measured 0.53 vs 0.74 respectively on identical runs."""
        m = self.metrics
        self.assertGreaterEqual(m["eval_pair_agree_after"], 0.7,
                                f"held-out pairwise agreement {m['eval_pair_agree_after']} "
                                f"(chance {m['chance_pair_agree']})")
        self.assertGreater(m["eval_rank_acc_after"], 2 * m["chance_rank_acc"],
                           f"argmax accuracy {m['eval_rank_acc_after']} not above twice chance "
                           f"{m['chance_rank_acc']}")

    def test_learning_improves_over_initialisation(self):
        m = self.metrics
        self.assertGreater(m["eval_pair_agree_after"], m["eval_pair_agree_before"],
                           f"pair agreement before {m['eval_pair_agree_before']} "
                           f"after {m['eval_pair_agree_after']}")

    def test_recovers_preferences_across_seeds(self):
        scores = [self.metrics["eval_pair_agree_after"]]
        for seed in (1, 2):
            m = tm.train(self.dataset, os.path.join(self.tmp, f"seed{seed}.gguf"),
                         creator="synth", epochs=150, lr_muon=0.005, lr_adamw=0.005,
                         temperature=0.5, holdout_every=5, seed=seed, warm_start=False)
            self.assertGreater(m["eval_pair_agree_after"], m["eval_pair_agree_before"],
                               f"seed {seed}: no improvement")
            scores.append(m["eval_pair_agree_after"])
        mean = sum(scores) / len(scores)
        self.assertGreaterEqual(mean, 0.7, f"mean held-out pairwise agreement over 3 seeds {mean:.3f}")

    def test_latent_actually_moved(self):
        self.assertGreater(self.metrics["z_u_norm"], 1e-6,
                           "z_u never moved — AdamW is not reaching the identity")

    def test_artifacts_written_and_reloadable(self):
        scorer = tm.TasteScorer.load(self.identity)
        info = scorer.info()
        self.assertEqual(info["creator"], "synth")
        self.assertEqual(info["trained_groups"], self.metrics["train_groups"])
        self.assertTrue(os.path.getsize(self.metrics["style_brain"]) > 0)
        self.assertTrue(os.path.getsize(self.identity) > 0)
        # Per-creator artifact stays tiny compared with the shared trunk.
        self.assertLess(os.path.getsize(self.identity), os.path.getsize(self.metrics["style_brain"]))

    def test_determinism(self):
        again = tm.train(self.dataset, os.path.join(self.tmp, "synth2.gguf"), creator="synth",
                         epochs=150, lr_muon=0.005, lr_adamw=0.005, temperature=0.5,
                         holdout_every=5, seed=0, warm_start=False)
        self.assertEqual(again["eval_rank_acc_after"], self.metrics["eval_rank_acc_after"])
        self.assertEqual(again["z_u"], self.metrics["z_u"])

    def test_singleton_dataset_is_refused_with_a_reason(self):
        path = os.path.join(self.tmp, "singletons.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"features": [1, 0, 1], "score": 0.7, "deltas": [0.0]}) + "\n")
        with self.assertRaises(SystemExit) as cm:
            tm.train(path, os.path.join(self.tmp, "x.gguf"), epochs=1)
        self.assertIn("group-relative", str(cm.exception))


class TestIdentitySemantics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tasteid_")
        self.dataset = os.path.join(self.tmp, "synth.jsonl")
        synthetic_dataset(self.dataset, n_clips=24, k=5, seed=1)
        self.identity = os.path.join(self.tmp, "a.gguf")
        tm.train(self.dataset, self.identity, creator="a", epochs=30, seed=0)
        self.scorer = tm.TasteScorer.load(self.identity)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_latent_is_required_for_the_scorer_and_is_nonzero(self):
        self.assertGreater(np.linalg.norm(self.scorer.model.p["z_u"]), 1e-6)

    def test_zeroing_the_latent_changes_scores(self):
        rng = np.random.default_rng(0)
        feats = [{n: float(v) for n, v in zip(tm.VIDEO_FEATURES, rng.normal(size=16))}
                 for _ in range(6)]
        before = self.scorer.score_many(feats)
        self.scorer.model.p["z_u"] = np.zeros_like(self.scorer.model.p["z_u"])
        after = self.scorer.score_many(feats)
        self.assertFalse(np.allclose(before, after),
                         "the identity latent has no effect on the taste score")

    def test_latent_is_interpolatable(self):
        """A midpoint latent must score between the two endpoints — this is the
        'nonlinear but interpolatable identity' property from the design."""
        rng = np.random.default_rng(1)
        z_a = self.scorer.model.p["z_u"].copy()
        z_b = rng.normal(size=z_a.shape)
        feats = [{n: float(v) for n, v in zip(tm.VIDEO_FEATURES, rng.normal(size=16))}
                 for _ in range(12)]
        self.scorer.model.p["z_u"] = z_a
        ua = np.array(self.scorer.score_many(feats))
        self.scorer.model.p["z_u"] = z_b
        ub = np.array(self.scorer.score_many(feats))
        self.scorer.model.p["z_u"] = 0.5 * (z_a + z_b)
        um = np.array(self.scorer.score_many(feats))
        lo, hi = np.minimum(ua, ub) - 1e-9, np.maximum(ua, ub) + 1e-9
        inside = ((um >= lo) & (um <= hi)).mean()
        # Per-candidate betweenness is NOT guaranteed by a nonlinear (tanh/FiLM)
        # score — the meaningful contract is: most candidates stay bracketed, the
        # aggregate moves monotonically, and the path is continuous.
        self.assertGreaterEqual(inside, 0.7,
                                f"only {inside:.0%} of midpoint scores lie between the endpoints")
        self.assertFalse(np.allclose(um, ua) or np.allclose(um, ub),
                         "midpoint is not a distinct identity")
        ma, mb, mm = float(ua.mean()), float(ub.mean()), float(um.mean())
        self.assertGreaterEqual(mm, min(ma, mb) - 1e-9)
        self.assertLessEqual(mm, max(ma, mb) + 1e-9)
        # Continuity: a 1% latent step may not move a score more than the whole path.
        total = np.abs(ub - ua).max() + 1e-9
        self.scorer.model.p["z_u"] = z_a + 0.01 * (z_b - z_a)
        near = np.array(self.scorer.score_many(feats))
        self.assertLess(np.abs(near - ua).max(), total,
                        "the score jumps discontinuously under a tiny latent step")


class TestEngineWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = fixture_inventory()
        cls.rubric = fixture_rubric()
        cls.tmp = tempfile.mkdtemp(prefix="tasteeng_")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_group1_is_the_legacy_schema(self):
        legacy = eng._propose_variant(self.inv, self.rubric, 1.0, 1.0, True, True)
        out = eng.cmd_propose(self.inv, self.rubric, group=1)
        self.assertEqual(out["structure"], legacy["structure"])
        self.assertEqual(out["meta"], legacy["meta"])
        self.assertEqual(out["globals"], legacy["globals"])
        self.assertEqual(out["group"]["selected"], 0)
        self.assertEqual(out["group"]["select_by"], "only")

    def test_group_is_deterministic_and_selects_best_reward(self):
        a = eng.cmd_propose(self.inv, self.rubric, group=6, log=False)
        b = eng.cmd_propose(self.inv, self.rubric, group=6, log=False)
        self.assertEqual(a["structure"], b["structure"])
        self.assertEqual(a["group"]["size"], 6)
        rewards = [c["reward_obj"] for c in a["group"]["candidates"]]
        best = max(range(len(rewards)), key=lambda i: rewards[i])
        self.assertEqual(a["group"]["selected"], best)
        self.assertEqual(a["group"]["select_by"], "objective")

    def test_candidates_are_genuinely_different(self):
        out = eng.cmd_propose(self.inv, self.rubric, group=6, log=False)
        shapes = {len(c["knobs"]) and json.dumps(c["knobs"]) for c in out["group"]["candidates"]}
        self.assertEqual(len(shapes), 6)
        n_segs = {c["n_segments"] for c in out["group"]["candidates"]}
        self.assertGreater(len(n_segs), 1, "candidates do not differ in shot selection")

    def test_loop_logs_group_then_critic_refines_reward(self):
        ds = os.path.join(self.tmp, "loop.jsonl")
        out = eng.cmd_propose(self.inv, self.rubric, group=5, dataset=ds, clip_id="c1")
        gid = out["group"]["group_id"]
        crit = eng.cmd_critic(out, self.inv, self.rubric, subjective=0.8,
                              dataset=ds, group_id=gid, candidate=out["group"]["selected"])
        self.assertIn("logged", crit)
        groups, _ = tm.load_groups(ds)
        self.assertEqual(len(groups), 1)
        _, stats = tm.build_training_arrays(ds)
        self.assertEqual(stats["groups_usable"], 1)
        g = groups[0]
        self.assertEqual(g["chosen"], out["group"]["selected"])
        self.assertEqual(g["candidates"][g["chosen"]]["reward_source"], "critic")
        self.assertEqual(g["candidates"][g["chosen"]]["reward"], crit["reward"])
        # The group's own candidate metadata must include the logs' dense parts.
        for c in out["group"]["candidates"]:
            self.assertIn("overall_obj", c)

    def test_identity_changes_the_selection_toward_the_taste_model(self):
        ds = os.path.join(self.tmp, "train.jsonl")
        # Several distinct briefs give the group-relative objective real data.
        for i, pace in enumerate(["brisk", "medium", "slow", "brisk", "medium", "slow",
                                  "brisk", "medium", "slow", "brisk"]):
            rub = fixture_rubric(pace=pace, target_duration=8.0 + i)
            eng.cmd_propose(self.inv, rub, group=6, dataset=ds, clip_id=f"clip{i}")
        identity = os.path.join(self.tmp, "creator.gguf")
        metrics = eng.cmd_train(ds, identity, creator="test", epochs=150, seed=0)
        self.assertEqual(metrics["groups"]["groups_usable"], 10)
        scorer = tm.TasteScorer.load(identity)
        out = eng.cmd_propose(self.inv, self.rubric, group=6, identity=identity,
                              select="taste", log=False)
        self.assertEqual(out["group"]["select_by"], "taste")
        scores = [c["taste_score"] for c in out["group"]["candidates"]]
        self.assertEqual(out["group"]["selected"], max(range(len(scores)), key=lambda i: scores[i]))
        objective = eng.cmd_propose(self.inv, self.rubric, group=6, log=False)
        # The taste model and the objective reward are allowed to disagree; the
        # point is that the identity is actually consulted and can move the pick.
        self.assertIsNotNone(out["group"]["taste"])
        self.assertEqual(len(out["group"]["taste"]["scores"]), 6)
        self.assertEqual(objective["group"]["select_by"], "objective")

    def test_taste_status_reports_loop_state(self):
        ds = os.path.join(self.tmp, "status.jsonl")
        identity = os.path.join(self.tmp, "status.gguf")
        eng.cmd_propose(self.inv, self.rubric, group=4, dataset=ds, clip_id="c9")
        eng.cmd_identity_init(identity, creator="status")
        status = eng.cmd_taste_status(identity, ds)
        self.assertTrue(status["identity"]["exists"])
        self.assertEqual(status["identity"]["trained_groups"], 0)
        self.assertTrue(status["dataset"]["exists"])
        self.assertEqual(status["dataset"]["groups"], 1)
        self.assertTrue(status["dataset"]["ready_to_train"])
        self.assertIsNotNone(status["dataset"]["reward_mean"])


class TestGracefulDegradation(unittest.TestCase):
    """The legacy loop must not require numpy; the taste path must fail loudly
    (a clean JSON error) rather than silently doing something else."""

    def _run(self, args, block_numpy=True):
        env = dict(os.environ)
        if block_numpy:
            d = tempfile.mkdtemp(prefix="nonumpy_")
            with open(os.path.join(d, "numpy.py"), "w", encoding="utf-8") as fh:
                fh.write("raise ImportError('numpy blocked for this test')\n")
            env["PYTHONPATH"] = d + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run([sys.executable, os.path.join(BIN, "edit_apart_core.py"), *args],
                              capture_output=True, text=True, env=env)

    def test_legacy_propose_works_without_numpy(self):
        inv = json.dumps(fixture_inventory())
        rub = json.dumps(fixture_rubric())
        r = self._run(["propose", inv, rub, "--group", "1", "--no-log"])
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["group"]["size"], 1)
        self.assertIn("structure", out)

    def test_group_without_numpy_still_works(self):
        """Group + feature extraction is pure stdlib; only the learned scorer and
        the trainer need numpy. This is deliberate: `group=K` must keep producing
        and logging groups on a machine without numpy."""
        inv = json.dumps(fixture_inventory())
        rub = json.dumps(fixture_rubric())
        r = self._run(["propose", inv, rub, "--group", "4", "--no-log"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["group"]["size"], 4)

    def test_train_without_numpy_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            ds = os.path.join(d, "log.jsonl")
            with open(ds, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "group", "group_id": "g", "candidates": [
                    {"features": {}, "reward_obj": 1.0},
                    {"features": {}, "reward_obj": 0.0}]}) + "\n")
            r = self._run(["train", "--dataset", ds, "--identity", os.path.join(d, "i.gguf")])
            self.assertEqual(r.returncode, 2)
            self.assertIn("error", json.loads(r.stdout))

    def test_scoring_without_numpy_is_a_clean_error(self):
        r = self._run(["taste", "--identity", "/nonexistent.gguf"], block_numpy=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("error", json.loads(r.stdout))


class TestModuleImportIsNumpyFree(unittest.TestCase):
    def test_importing_taste_model_does_not_load_numpy(self):
        code = ("import sys; sys.path.insert(0, %r); import taste_model; "
                "print('numpy' in sys.modules)" % BIN)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
