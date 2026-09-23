#!/usr/bin/env python3
"""Photo (single-image) taste tests: features, grouped proposal, loop wiring, and
the per-creator identity demonstration.

Photo is the harder modality: there is no temporal axis, so

  * the feature layout has to SYNTHESISE spatial structure (the crop's geometry
    and the region grid it keeps) where video leans on the shot inventory,
  * the objective reward is defined on a RENDERED image, so scoring a candidate
    group costs a render per candidate, and
  * the reward is partly QUANTISED (a tolerance on mean luma), which makes
    near-duplicate candidates tie and gives the group-relative update exactly
    zero signal. That is asserted here rather than assumed: a group must not be
    a total tie, or there is nothing for GRPO to learn from.

Requires ImageMagick for the grouped/render paths; the feature tests do not.
Run:  <venv>/bin/python tests/test_photo_taste.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "bin")
PHOTO_CORE = os.path.join(BIN, "photo_core.py")
sys.path.insert(0, BIN)

import numpy as np  # noqa: E402

import photo_core as pc  # noqa: E402
import taste_model as tm  # noqa: E402

PY = os.environ.get("DSH_EDIT_PY", sys.executable)
MAGICK = shutil.which("magick") or shutil.which("convert")
HAVE_MAGICK = MAGICK is not None


def engine(*args) -> dict:
    r = subprocess.run([PY, PHOTO_CORE, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"photo engine {args[0]} failed: {r.stdout} {r.stderr}")
    return json.loads(r.stdout)


def engine_env(env: dict, *args) -> dict:
    """Run the engine with a custom environment (toolchain overrides)."""
    r = subprocess.run([PY, PHOTO_CORE, *args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"photo engine {args[0]} failed: {r.stdout} {r.stderr}")
    return json.loads(r.stdout)


def make_image(path: str) -> str:
    """A deterministic image with real spatial structure: a bright highlight
    upper-right, a dark block lower-left, a mid band, and a warm block lower-right.

    Built from `xc:` + shapes rather than `gradient:`, because the gradient coder
    is not present in every ImageMagick build (this one has none).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [MAGICK, "-size", "1600x1200", "xc:#232c38",
           "-fill", "#6b7684", "-draw", "rectangle 0,300 1600,560",
           "-fill", "#ffe9a8", "-draw", "circle 1150,420 1150,240",
           "-fill", "#0d1218", "-draw", "rectangle 100,860 700,1150",
           "-fill", "#8a5a2b", "-draw", "rectangle 900,780 1450,1160",
           path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"test image generation failed: {r.stderr[:300]}")
    return path


def brief(target_luma: float = 0.52, crop=(0.1, 0.1, 0.6, 0.5), **over) -> dict:
    r = {"intent": "warm punch", "saturation": 1.15, "crop": {"percent": list(crop)},
         "width": 1200, "target_luma": target_luma}
    r.update(over)
    return r


def crop_area_of(operations: list[dict], inspect: dict) -> float:
    w = float(inspect.get("width") or 1.0)
    h = float(inspect.get("height") or 1.0)
    for op in operations:
        if op.get("op") == "crop":
            return (float(op["w"]) * float(op["h"])) / (w * h)
    return 1.0


# ---------------------------------------------------------------------------
# features (no renderer needed)
# ---------------------------------------------------------------------------
class TestPhotoFeatures(unittest.TestCase):
    def setUp(self):
        self.inspect = {
            "source": "synthetic.png", "width": 1600, "height": 1200,
            "mean_luma": 0.40, "luma_std": 0.18, "saturation": 0.2,
            # top-right = highlight, bottom-left = shadow, as the tests assume
            "regions": {"grid": 4, "cells": [
                [{"luma": 0.10, "sat": 0.05}, {"luma": 0.12, "sat": 0.05},
                 {"luma": 0.85, "sat": 0.30}, {"luma": 0.88, "sat": 0.32}],
                [{"luma": 0.11, "sat": 0.05}, {"luma": 0.14, "sat": 0.06},
                 {"luma": 0.80, "sat": 0.28}, {"luma": 0.82, "sat": 0.29}],
                [{"luma": 0.20, "sat": 0.08}, {"luma": 0.22, "sat": 0.09},
                 {"luma": 0.30, "sat": 0.12}, {"luma": 0.32, "sat": 0.13}],
                [{"luma": 0.05, "sat": 0.03}, {"luma": 0.06, "sat": 0.03},
                 {"luma": 0.25, "sat": 0.10}, {"luma": 0.28, "sat": 0.11}],
            ]},
        }

    def test_layout_is_declared_and_stable(self):
        f = tm.features_photo({"operations": []}, self.inspect, brief())
        self.assertEqual(sorted(f), sorted(tm.PHOTO_FEATURES))
        self.assertEqual(len(tm.to_vector(f, tm.FEATURE_SPEC_PHOTO)), len(tm.PHOTO_FEATURES))
        again = tm.features_photo({"operations": []}, self.inspect, brief())
        self.assertEqual(f, again)

    def test_full_frame_crop_defaults(self):
        f = tm.features_photo({"operations": []}, self.inspect, brief())
        self.assertEqual(f["has_crop"], 0.0)
        self.assertAlmostEqual(f["crop_area"], 1.0, places=6)
        self.assertAlmostEqual(f["crop_centre_dev"], 0.0, places=6)

    def test_crop_geometry_is_measured_not_guessed(self):
        # Centred crop: zero centre deviation, but the frame centre is maximally
        # far from every rule-of-thirds point (equidistant from all four).
        crop = {"op": "crop", "x": 400, "y": 300, "w": 800, "h": 600}
        f = tm.features_photo({"operations": [crop]}, self.inspect, brief())
        self.assertAlmostEqual(f["crop_area"], 0.25, places=6)  # 800*600/(1600*1200)
        self.assertAlmostEqual(f["crop_centre_dev"], 0.0, places=6)
        self.assertAlmostEqual(f["crop_aspect_dev"], 0.0, places=6)
        self.assertAlmostEqual(f["crop_thirds_dist"], 1 / 3, places=2)
        # A crop whose centre sits ON a thirds intersection scores zero distance.
        # (1600/3, 1200/3) = (533, 400); half-size 200x180 keeps the centre there.
        on_third = {"op": "crop", "x": 333, "y": 220, "w": 400, "h": 360}
        t = tm.features_photo({"operations": [on_third]}, self.inspect, brief())
        self.assertLess(t["crop_thirds_dist"], 0.02)
        # A corner crop deviates from centre and from the thirds.
        off = {"op": "crop", "x": 0, "y": 0, "w": 800, "h": 600}
        g = tm.features_photo({"operations": [off]}, self.inspect, brief())
        self.assertGreater(g["crop_centre_dev"], 0.2)
        self.assertGreater(g["crop_thirds_dist"], 0.1)

    def test_crop_region_statistics_follow_the_grid(self):
        """A crop that keeps the bright highlight must report higher luma,
        contrast and salience than one that keeps only the dark corner."""
        bright = {"op": "crop", "x": 800, "y": 0, "w": 800, "h": 600}
        dark = {"op": "crop", "x": 0, "y": 900, "w": 400, "h": 300}
        fb = tm.features_photo({"operations": [bright]}, self.inspect, brief())
        fd = tm.features_photo({"operations": [dark]}, self.inspect, brief())
        self.assertGreater(fb["crop_luma"], fd["crop_luma"])
        self.assertGreater(fb["crop_salience"], fd["crop_salience"])
        self.assertGreater(fb["crop_contrast"], fd["crop_contrast"])

    def test_scale_crop_keeps_centre_and_scales_area(self):
        rect = {"x": 400, "y": 300, "w": 800, "h": 600}
        smaller = pc._scale_crop(rect, 0.5, self.inspect)
        bigger = pc._scale_crop(rect, 2.0, self.inspect)
        for out, factor in ((smaller, 0.5), (bigger, 2.0)):
            self.assertAlmostEqual((out["w"] * out["h"]) / (rect["w"] * rect["h"]),
                                   factor, places=1)
            self.assertAlmostEqual(out["x"] + out["w"] / 2, 800, delta=2)
            self.assertAlmostEqual(out["y"] + out["h"] / 2, 600, delta=2)
        # clamped to the frame
        self.assertLessEqual(bigger["x"] + bigger["w"], self.inspect["width"])
        self.assertGreaterEqual(bigger["x"], 0)

    def test_grade_and_caption_features(self):
        ops = [{"op": "grade", "exposure": 1.3, "saturation": 1.0, "contrast": 0.4,
                "temperature": -0.2},
               {"op": "overlay_text", "text": "x" * 40, "size": 60},
               {"op": "resize", "width": 800},
               {"op": "sharpen", "amount": 1.5}]
        f = tm.features_photo({"operations": ops}, self.inspect, brief())
        self.assertAlmostEqual(f["exposure_dev"], 0.3, places=6)
        self.assertAlmostEqual(f["sat_dev"], 0.0, places=6)
        self.assertAlmostEqual(f["contrast_amt"], 0.4, places=6)
        self.assertAlmostEqual(f["temp_abs"], 0.2, places=6)
        self.assertAlmostEqual(f["sharpen_rel"], 1.5, places=6)
        self.assertAlmostEqual(f["width_ratio"], 0.5, places=6)
        self.assertGreater(f["caption_len"], 0.0)
        self.assertAlmostEqual(f["caption_size_rel"], 60 / 1600, places=6)

    def test_predicted_luma_uses_the_crop_region(self):
        dark = {"op": "crop", "x": 0, "y": 900, "w": 400, "h": 300}
        bright = {"op": "crop", "x": 800, "y": 0, "w": 800, "h": 600}
        g = {"op": "grade", "exposure": 1.0, "saturation": 1.0, "contrast": 0.0,
             "temperature": 0.0}
        # target 0.52; the bright crop sits closer to it than the dark one
        fb = tm.features_photo({"operations": [bright, g]}, self.inspect, brief(0.52))
        fd = tm.features_photo({"operations": [dark, g]}, self.inspect, brief(0.52))
        self.assertLess(fb["pred_luma_err"], fd["pred_luma_err"])

    def test_missing_region_grid_degrades_to_zeros(self):
        bare = {k: v for k, v in self.inspect.items() if k != "regions"}
        f = tm.features_photo({"operations": [{"op": "crop", "x": 0, "y": 0,
                                               "w": 400, "h": 300}]}, bare, brief())
        self.assertEqual(f["crop_luma"], 0.0)
        self.assertEqual(f["crop_salience"], 0.0)
        self.assertEqual(len(tm.to_vector(f, tm.FEATURE_SPEC_PHOTO)), len(tm.PHOTO_FEATURES))

    def test_features_for_dispatches_by_spec(self):
        f = tm.features_for(tm.FEATURE_SPEC_PHOTO, {"operations": []}, self.inspect, brief())
        self.assertEqual(sorted(f), sorted(tm.PHOTO_FEATURES))
        with self.assertRaises(ValueError):
            tm.features_for("photo/v9", {"operations": []}, self.inspect, brief())


@unittest.skipUnless(HAVE_MAGICK, "ImageMagick not available")
class TestImageMagickCompatibility(unittest.TestCase):
    """ImageMagick 6 ships `convert` (renderer) and `identify` (prober) as two
    binaries, and `convert identify …` is not a thing. Resolving only `convert`
    and asking it for dimensions used to return 0x0 SILENTLY, which turned every
    percent crop into `0x0+0+0` and zeroed the whole spatial feature axis while
    every downstream step still reported success."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="photoim_")
        cls.img = make_image(os.path.join(cls.tmp, "synthetic.png"))
        cls.inspect = engine("inspect", cls.img)
        cls.rubric = brief()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_identify_prefix_is_not_a_convert_subcommand(self):
        original = pc.MAGICK
        try:
            pc.MAGICK = "/usr/bin/convert"          # IM6-style renderer
            prefix = pc._identify_prefix()
        finally:
            pc.MAGICK = original
        self.assertNotEqual(prefix, ["/usr/bin/convert", "identify"])
        self.assertEqual(os.path.basename(prefix[0]), "identify")

    def test_identify_prefix_uses_the_magick_subcommand_on_im7(self):
        original = pc.MAGICK
        try:
            pc.MAGICK = "/usr/bin/magick"
            prefix = pc._identify_prefix()
        finally:
            pc.MAGICK = original
        self.assertEqual(prefix, ["/usr/bin/magick", "identify"])

    @unittest.skipUnless(os.path.exists("/usr/bin/convert"), "no IM6-style convert binary")
    def test_inspect_returns_real_dimensions_with_a_convert_only_renderer(self):
        env = dict(os.environ, DSH_IMAGEMAGICK="/usr/bin/convert")
        env.pop("DSH_IDENTIFY", None)
        out = engine_env(env, "inspect", self.img)
        self.assertGreater(out["width"], 0, "0x0 dims silently degenerate every crop")
        self.assertGreater(out["height"], 0)
        self.assertEqual(out["width"], 1600)

    def test_dimension_failure_is_loud_not_zero(self):
        env = dict(os.environ, DSH_IMAGEMAGICK=MAGICK, DSH_IDENTIFY="/bin/false")
        r = subprocess.run([PY, PHOTO_CORE, "inspect", self.img],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2, "a broken prober must not pass silently")
        self.assertIn("dimensions", json.loads(r.stdout)["error"])

    def test_propose_refuses_a_dimension_less_inspect(self):
        blind = {**self.inspect, "width": 0, "height": 0}
        with self.assertRaises(RuntimeError) as cm:
            pc.cmd_propose(self.img, blind, self.rubric)
        self.assertIn("dimensions", str(cm.exception))

    def test_selftest_reports_a_working_environment(self):
        out = engine("selftest")
        self.assertTrue(out["ok"])
        self.assertEqual(out["dims"], "600x400")
        self.assertGreater(out["grid_luma_spread"], 0.02,
                           "a degenerate region grid means composition is invisible")
        self.assertGreaterEqual(len(set(out["candidate_crop_areas"])), 2)

    def test_selftest_fails_loudly_with_a_broken_prober(self):
        env = dict(os.environ, DSH_IMAGEMAGICK="/usr/bin/convert", DSH_IDENTIFY="/bin/false")
        r = subprocess.run([PY, PHOTO_CORE, "selftest"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("dimensions", json.loads(r.stdout)["error"])

    def test_temperature_reaches_the_render(self):
        """`-colorize 0` was a no-op, so a declared+featured field was dead."""
        d = tempfile.mkdtemp(prefix="phototemp_")
        try:
            def grade(t):
                return {"operations": [{"op": "grade", "exposure": 1.0, "contrast": 0.0,
                                        "saturation": 1.0, "temperature": t}]}
            outs = {}
            for label, t in (("neutral", 0.0), ("warm", 0.4)):
                path = os.path.join(d, f"{label}.png")
                pc.cmd_render(self.img, grade(t), path)
                outs[label] = pc.cmd_inspect(path)
            rb = lambda i: i["mean_rgb"][0] - i["mean_rgb"][2]
            self.assertGreater(rb(outs["warm"]), rb(outs["neutral"]),
                               "a positive temperature must warm the render (R up, B down)")
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# reward grading (no renderer: the inspected metrics are injected)
# ---------------------------------------------------------------------------
class TestPhotoRewardGrading(unittest.TestCase):
    """The objective reward used to be a hard threshold, which made every
    candidate that missed it score identically (a tied group = zero GRPO signal).
    Grading must REFINE it: identical for candidates that pass, informative for
    those that only just miss."""

    def _score(self, mean_luma: float, luma_std: float, target: float = 0.5,
               tol: float = 0.15):
        original = pc.cmd_inspect
        pc.cmd_inspect = lambda _p: {
            "source": "x", "width": 800, "height": 600, "mean_luma": mean_luma,
            "luma_std": luma_std, "lum_p50": mean_luma, "lum_p95": mean_luma,
            "saturation": 0.2, "mean_rgb": [0.2, 0.2, 0.2], "regions": {"grid": 0, "cells": []},
        }
        try:
            return pc._score_objective({"operations": []}, {"target_luma": target,
                                                            "exposure_tol": tol}, "x", None)
        finally:
            pc.cmd_inspect = original

    def test_passing_candidates_score_exactly_as_before(self):
        out = self._score(mean_luma=0.50, luma_std=0.20)   # inside tol, contrast ok
        self.assertTrue(out["metrics"]["exposure_ok"])
        self.assertEqual(out["metrics"]["exposure_score"], 1.0)
        self.assertEqual(out["metrics"]["contrast_score"], 1.0)

    def test_a_near_miss_keeps_partial_credit(self):
        inside = self._score(mean_luma=0.50, luma_std=0.20)
        near = self._score(mean_luma=0.50 + 0.15 + 0.05, luma_std=0.20)  # 0.05 past tol
        far = self._score(mean_luma=0.50 + 0.40, luma_std=0.20)          # 0.25 past tol
        self.assertFalse(near["metrics"]["exposure_ok"])
        self.assertGreater(near["metrics"]["exposure_score"], 0.0)
        self.assertLess(near["metrics"]["exposure_score"], 1.0)
        self.assertEqual(far["metrics"]["exposure_score"], 0.0)
        self.assertGreater(near["overall_score"], far["overall_score"])
        self.assertGreater(inside["overall_score"], near["overall_score"])

    def test_partial_credit_is_monotone_in_the_distance(self):
        scores = [self._score(mean_luma=0.5 + 0.15 + d, luma_std=0.20)["metrics"]["exposure_score"]
                  for d in (0.01, 0.05, 0.10, 0.14, 0.20)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores))

    def test_contrast_shortfall_is_graded(self):
        ok = self._score(mean_luma=0.5, luma_std=0.20)
        short = self._score(mean_luma=0.5, luma_std=0.06)   # half of min_contrast 0.12
        self.assertTrue(ok["metrics"]["contrast_ok"])
        self.assertFalse(short["metrics"]["contrast_ok"])
        self.assertAlmostEqual(short["metrics"]["contrast_score"], 0.5, places=2)
        self.assertLess(short["overall_score"], ok["overall_score"])


# ---------------------------------------------------------------------------
# grouped proposal (needs the renderer: candidates are scored on renders)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_MAGICK, "ImageMagick not available")
class TestPhotoGroupedPropose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="phototaste_")
        cls.img = make_image(os.path.join(cls.tmp, "synthetic.png"))
        cls.inspect = engine("inspect", cls.img)
        cls.rubric = brief()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_inspect_carries_a_spatial_grid(self):
        regions = self.inspect.get("regions") or {}
        self.assertEqual(regions.get("grid"), 4)
        cells = regions["cells"]
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(len(row) == 4 for row in cells))
        lumas = [c["luma"] for row in cells for c in row]
        self.assertGreater(max(lumas) - min(lumas), 0.1,
                           "the synthetic image has no spatial structure to measure")
        # global stats must be untouched by adding the grid
        self.assertIn("mean_luma", self.inspect)
        self.assertIn("luma_std", self.inspect)

    def test_group1_is_the_legacy_schema_without_rendering(self):
        legacy = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric))
        out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric),
                     "--group", "1")
        self.assertEqual(out["operations"], legacy["operations"])
        self.assertEqual(out["group"]["size"], 1)
        self.assertEqual(out["group"]["select_by"], "only")
        self.assertEqual(out["group"]["candidates"][0]["reward_obj"], None,
                         "group=1 with no dataset must not render")

    def test_group_candidates_are_distinct_and_include_the_legacy_ops(self):
        legacy = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric))
        out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric),
                     "--group", "8", "--no-log")
        ops = [json.dumps(c["operations"], sort_keys=True) for c in out["group"]["candidates"]]
        self.assertEqual(len(set(ops)), len(ops), "duplicate candidates in the group")
        self.assertEqual(out["group"]["candidates"][0]["operations"], legacy["operations"])
        self.assertEqual(out["group"]["size"], 8)

    def test_group_is_not_a_total_tie(self):
        """The quantisation regression guard: a thresholded reward made every
        candidate score identically (spread 0.0), which is zero GRPO signal."""
        ds = os.path.join(self.tmp, "tie.jsonl")
        out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric),
                     "--group", "8", "--dataset", ds, "--clip-id", "tie")
        rewards = [c["reward_obj"] for c in out["group"]["candidates"]]
        self.assertIsNotNone(rewards[0], "candidates were not scored")
        self.assertGreater(len(set(rewards)), 1, f"all candidates tied at {rewards[0]}")
        rec = json.loads(open(ds, encoding="utf-8").read().strip())
        self.assertFalse(rec["tie"])
        self.assertGreater(rec["reward_spread"], 0.0)
        self.assertEqual(rec["feature_spec"], tm.FEATURE_SPEC_PHOTO)

    def test_candidate_renders_exist_and_are_critiqued(self):
        ds = os.path.join(self.tmp, "renders.jsonl")
        out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric),
                     "--group", "4", "--dataset", ds, "--clip-id", "r")
        for cand in out["group"]["candidates"]:
            self.assertTrue(cand["render"] and os.path.exists(cand["render"]),
                            f"candidate {cand['idx']} was not rendered")
            self.assertIsInstance(cand["reward_obj"], (int, float))
            self.assertIsInstance(cand["overall_obj"], (int, float))

    def test_selection_is_the_objective_argmax_without_an_identity(self):
        out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric),
                     "--group", "6", "--no-log")
        rewards = [c["reward_obj"] for c in out["group"]["candidates"]]
        best = max(range(len(rewards)), key=lambda i: rewards[i])
        self.assertEqual(out["group"]["selected"], best)
        self.assertEqual(out["group"]["select_by"], "objective")

    def test_nudges_never_override_an_explicit_rubric_value(self):
        explicit = brief(exposure=0.9, contrast=0.2)
        ops = [op for op in pc._legacy_ops(self.img, self.inspect, explicit)
               if op.get("op") == "grade"]
        self.assertTrue(ops)
        for cand in engine("propose", self.img, json.dumps(self.inspect),
                           json.dumps(explicit), "--group", "8", "--no-log")["group"]["candidates"]:
            grade = [o for o in cand["operations"] if o.get("op") == "grade"]
            for g in grade:
                # exposure/contrast may be scaled by the grade factor, but the
                # nudge must not have been added on top of an explicit value
                self.assertLessEqual(g["contrast"], 0.2 + 1e-9)

    def test_taste_selection_requires_an_identity(self):
        r = subprocess.run([PY, PHOTO_CORE, "propose", self.img, json.dumps(self.inspect),
                            json.dumps(self.rubric), "--group", "4", "--select", "taste"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("identity", json.loads(r.stdout)["error"])


# ---------------------------------------------------------------------------
# loop wiring: logging, folding, training, identity selection
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_MAGICK, "ImageMagick not available")
class TestPhotoLoopWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="photoloop_")
        cls.img = make_image(os.path.join(cls.tmp, "synthetic.png"))
        cls.inspect = engine("inspect", cls.img)
        cls.rubric = brief()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _corpus(self, name: str, briefs: list[dict], tightest: bool) -> str:
        ds = os.path.join(self.tmp, f"{name}.jsonl")
        for i, rub in enumerate(briefs):
            out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(rub),
                         "--group", "8", "--dataset", ds, "--clip-id", f"{name}{i}")
            cands = out["group"]["candidates"]
            picker = min if tightest else max
            pick = picker(range(len(cands)),
                          key=lambda j: crop_area_of(cands[j]["operations"], self.inspect))
            crit = engine("critic", json.dumps({"operations": cands[pick]["operations"]}),
                          json.dumps(rub), cands[pick]["render"], "--dataset", ds,
                          "--group-id", out["group"]["group_id"],
                          "--candidate", str(pick), "--chosen-by", "creator")
            self.assertIn("logged", crit)
        return ds

    def test_critic_reward_is_logged_as_a_revealed_preference(self):
        ds = self._corpus("log", [self.rubric], tightest=True)
        groups, stats = tm.load_groups(ds)
        self.assertEqual(len(groups), 1)
        self.assertEqual(stats["reward_records"], 1)
        g = groups[0]
        self.assertEqual(g["chosen_by"], "creator")
        self.assertIsNotNone(g["chosen"])
        prepared, pstats = tm.build_training_arrays(ds)
        self.assertEqual(pstats["groups_usable"], 1)
        p = prepared[0]
        self.assertEqual(int(p["rewards"].argmax()), p["chosen"],
                         "the revealed pick did not become the group's top reward")
        self.assertEqual(p["spec"], tm.FEATURE_SPEC_PHOTO)

    def test_identity_init_and_taste_selection_on_real_features(self):
        ds = self._corpus("sel", [brief(target_luma=0.45 + 0.02 * i) for i in range(6)],
                          tightest=True)
        identity = os.path.join(self.tmp, "photo_identity.gguf")
        eng = engine("identity_init", "--identity", identity, "--creator", "photographer",
                     "--spec", tm.FEATURE_SPEC_PHOTO)
        self.assertTrue(os.path.exists(identity))
        self.assertEqual(eng["feature_spec"], tm.FEATURE_SPEC_PHOTO)
        metrics = engine("taste_train", "--dataset", ds, "--identity", identity,
                         "--creator", "photographer", "--epochs", "40")
        self.assertEqual(metrics["feature_spec"], tm.FEATURE_SPEC_PHOTO)
        self.assertEqual(metrics["dims"]["d_in"], len(tm.PHOTO_FEATURES))
        # a photo style-brain is a separate artifact from the video one
        self.assertIn("photo_v1", metrics["style_brain"])
        scorer = tm.TasteScorer.load(identity)
        self.assertEqual(scorer.feature_spec, tm.FEATURE_SPEC_PHOTO)
        out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(self.rubric),
                     "--group", "6", "--identity", identity, "--select", "taste", "--no-log")
        self.assertEqual(out["group"]["select_by"], "taste")
        scores = [c["taste_score"] for c in out["group"]["candidates"]]
        self.assertTrue(all(s is not None for s in scores))
        self.assertEqual(out["group"]["selected"],
                         max(range(len(scores)), key=lambda i: scores[i]))

    def test_status_reports_photo_groups(self):
        ds = self._corpus("status", [self.rubric], tightest=False)
        identity = os.path.join(self.tmp, "status_identity.gguf")
        engine("identity_init", "--identity", identity, "--creator", "s", "--spec", tm.FEATURE_SPEC_PHOTO)
        status = engine("taste_status", "--identity", identity, "--dataset", ds)
        self.assertTrue(status["identity"]["exists"])
        self.assertEqual(status["identity"]["feature_spec"], tm.FEATURE_SPEC_PHOTO)
        self.assertEqual(status["dataset"]["groups"], 1)
        self.assertTrue(status["dataset"]["ready_to_train"])


# ---------------------------------------------------------------------------
# the demonstration: one shared trunk, two creators with opposite spatial taste
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_MAGICK, "ImageMagick not available")
class TestPhotoPerCreatorIdentity(unittest.TestCase):
    """One shared trunk, two creators with opposite SPATIAL taste (tight vs loose
    framing), measured on held-out neutral briefs.

    Unlike video, this modality's taste variable (crop geometry) is confounded
    with the image content the crop reveals, so the same measurement is taken
    twice: on briefs from the training family, and on briefs with an unseen crop
    geometry. The second is a documented limitation, not a result to hide.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="photoe2e_")
        cls.img = make_image(os.path.join(cls.tmp, "synthetic.png"))
        cls.inspect = engine("inspect", cls.img)

        def family(i):
            return (0.05 + 0.01 * (i % 3), 0.08, 0.55 + 0.02 * (i % 4), 0.45)

        cls.train_briefs = [brief(target_luma=0.42 + 0.02 * i, crop=family(i))
                            for i in range(18)]
        cls.neutral = [brief(target_luma=0.46 + 0.01 * i, crop=family(i)) for i in range(10)]
        # both eval sets are distinct-input sets (target_luma varies per brief)
        # DISTINCT target_luma per brief: ten identical briefs would be one
        # distinct input evaluated ten times, which is not an n of ten and cannot
        # support a "0/10" style claim. (It still shares ONE unseen crop geometry —
        # a limitation this suite states rather than hides.)
        cls.ood = [brief(target_luma=0.44 + 0.02 * i, crop=(0.1, 0.1, 0.6, 0.5))
                   for i in range(10)]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _corpus(self, name: str, tightest: bool) -> str:
        ds = os.path.join(self.tmp, f"{name}.jsonl")
        for i, rub in enumerate(self.train_briefs):
            out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(rub),
                         "--group", "8", "--dataset", ds, "--clip-id", f"{name}{i}",
                         "--workdir", os.path.join(self.tmp, f"w_{name}_{i}"))
            cands = out["group"]["candidates"]
            picker = min if tightest else max
            pick = picker(range(len(cands)),
                          key=lambda j: crop_area_of(cands[j]["operations"], self.inspect))
            engine("critic", json.dumps({"operations": cands[pick]["operations"]}),
                   json.dumps(rub), cands[pick]["render"], "--dataset", ds,
                   "--group-id", out["group"]["group_id"],
                   "--candidate", str(pick), "--chosen-by", "creator")
        return ds

    def _train_pair(self):
        if getattr(self.__class__, "_trained", False):
            return
        cls = self.__class__
        ds_tight = cls._corpus(cls, "tight", tightest=True)
        ds_loose = cls._corpus(cls, "loose", tightest=False)
        union = os.path.join(cls.tmp, "union.jsonl")
        with open(union, "w", encoding="utf-8") as out:
            for p in (ds_tight, ds_loose):
                with open(p, encoding="utf-8") as fh:
                    out.write(fh.read())
        style = os.path.join(cls.tmp, "style_brain_photo_v1.gguf")
        tm.train(union, os.path.join(cls.tmp, "general.gguf"), style=style,
                 creator="general", epochs=150, seed=0)
        ident = {}
        for name, ds in (("tight", ds_tight), ("loose", ds_loose)):
            path = os.path.join(cls.tmp, f"{name}.gguf")
            m = tm.train(ds, path, style=style, creator=name, epochs=300, seed=0,
                         freeze_style=True)
            assert m["style_frozen"]
            ident[name] = path
        digests = {tm.TasteScorer.load(p, style=style).model.shared_digest()
                   for p in ident.values()}
        assert len(digests) == 1, "per-creator runs changed the shared style-brain"
        cls._trained = True
        cls._style = style
        cls._ident = ident
        cls._ds_tight, cls._ds_loose = ds_tight, ds_loose

    def _picks(self, briefs: list[dict]) -> dict:
        self._train_pair()
        cached = getattr(self.__class__, "_picks_cache", {})
        key = id(briefs)
        if key in cached:
            return cached[key]
        picks = {"tight": [], "loose": []}
        for k, rub in enumerate(briefs):
            for name in ("tight", "loose"):
                out = engine("propose", self.img, json.dumps(self.inspect), json.dumps(rub),
                             "--group", "8", "--identity", self._ident[name],
                             "--style", self._style, "--select", "taste", "--no-log",
                             "--workdir", os.path.join(self.tmp, f"eval_{name}_{k}"))
                g = out["group"]
                self.assertEqual(g["select_by"], "taste")
                chosen = g["candidates"][g["selected"]]
                picks[name].append(crop_area_of(chosen["operations"], self.inspect))
        cached[key] = picks
        self.__class__._picks_cache = cached
        return picks

    def test_two_creators_on_one_shared_trunk_frame_differently(self):
        picks = self._picks(self.neutral)
        a, b = picks["tight"], picks["loose"]
        mean_tight = sum(a) / len(a)
        mean_loose = sum(b) / len(b)
        differing = sum(1 for x, y in zip(a, b) if abs(x - y) > 1e-9)
        tighter = sum(1 for x, y in zip(a, b) if x < y)
        print(f"\n  [in-distribution] neutral briefs: {len(self.neutral)}")
        print(f"  mean selected crop area  tight={mean_tight:.3f}  loose={mean_loose:.3f}")
        print(f"  briefs picked differently: {differing}/{len(self.neutral)}"
              f" (tight chose tighter in {tighter})")
        print(f"  tight picks: {[round(x, 3) for x in a]}")
        print(f"  loose picks: {[round(x, 3) for x in b]}")
        self.assertGreaterEqual(differing, len(self.neutral) // 2,
                                "the per-creator identity barely affected the selection")
        self.assertLess(mean_tight, mean_loose,
                        "the tight-crop identity did not learn to frame tighter")
        self.assertGreaterEqual(tighter, len(self.neutral) // 2)

    def test_preference_transfers_to_an_unseen_geometry_but_more_weakly(self):
        """Re-measured after the GRPO ratio baseline was corrected.

        The earlier "no transfer" result (0/10) was produced by an objective whose
        negative advantages had no gradient (the ratio was a constant 1/Z). With
        that fixed the preference DOES transfer to an unseen crop geometry — but
        with a smaller margin than in-distribution, so the geometry/content
        confound documented in docs/paper-findings.md (Finding 4c) narrows the
        effect rather than erasing it.
        """
        ood = self._picks(self.ood)
        in_dist = self._picks(self.neutral)
        ood_diff = sum(1 for x, y in zip(ood["tight"], ood["loose"]) if abs(x - y) > 1e-9)
        in_diff = sum(1 for x, y in zip(in_dist["tight"], in_dist["loose"]) if abs(x - y) > 1e-9)

        def gap(picks):
            mean = lambda xs: sum(xs) / len(xs)
            return mean(picks["loose"]) - mean(picks["tight"])

        ood_gap, in_gap = gap(ood), gap(in_dist)
        print(f"\n  [out-of-distribution] ONE unseen crop geometry, {len(self.ood)} distinct briefs")
        print(f"  briefs picked differently: {ood_diff}/{len(self.ood)}"
              f" | margin (loose-tight) {ood_gap:.3f} vs in-distribution {in_gap:.3f}")
        print(f"  tight picks: {[round(x, 3) for x in ood['tight']]}")
        print(f"  loose picks: {[round(x, 3) for x in ood['loose']]}")
        self.assertGreaterEqual(ood_diff, len(self.ood) // 2,
                                "the preference no longer transfers to the unseen geometry")
        self.assertLess(ood_gap, in_gap,
                        "the unseen geometry should narrow the margin, not widen it")

    def test_training_groups_are_not_all_ties(self):
        self._train_pair()
        groups, _ = tm.load_groups(self._ds_tight)
        spreads = []
        for g in groups:
            rewards = [c["reward"] for c in g["candidates"]]
            spreads.append(max(rewards) - min(rewards))
        informative = sum(1 for s in spreads if s > 0)
        print(f"\n  [photo groups] non-zero reward spread: {informative}/{len(groups)}")
        self.assertGreaterEqual(informative, max(1, len(groups) // 2),
                                "most photo groups carry no group-relative signal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
