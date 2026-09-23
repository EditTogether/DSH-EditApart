#!/usr/bin/env python3
"""End-to-end test of the wired-in taste loop on REAL footage.

This is the test that answers "is the identity learning actually part of the
editor, or still a side program?":

  1. build an enriched shot inventory from a real video (scenedetect + ffmpeg),
  2. drive the real loop over two creators who reveal different tastes: when a
     group of rubric-equivalent candidates is proposed, the "long-take" creator
     renders the longest candidate and the "short-take" creator the shortest.
     The critic's dense reward is logged for the rendered pick, and the pick is
     logged with `chosen_by="creator"` so it enters the objective as a REVEALED
     preference — a rubric-derived reward alone cannot identify per-user taste
     (ablation: 5/12 briefs differing with the direction at chance, versus 12/12
     with the preference term; see docs/paper-findings.md),
  3. train ONE shared style-brain on the union, then train each creator's
     identity with `freeze_style=True` (only `z_u` moves; the trunk is shared),
  4. ask both identities to edit the SAME neutral brief and compare what the loop
     actually selects,
  5. render + critique + revise the taste-selected schema with real ffmpeg to
     confirm the existing tool loop still runs end to end.

Asserted: the loop logs and folds every group with its revealed pick; both
identities train and load; the shared trunk is byte-identical after both
per-creator runs; scoring is deterministic; selection equals the taste argmax;
and on held-out neutral briefs the two identities pick different candidates, in
the predicted direction (the long-take identity selects materially longer edits).
Measured on the reference footage: 12/12 briefs differed, 12/12 in the expected
direction, 24.9s vs 12.0s mean selected duration.

Requirements: a video via DSH_EDITAPART_E2E_SRC, or an already-enriched
inventory via DSH_EDITAPART_E2E_INVENTORY (skips otherwise). ffmpeg/scenedetect
come from the same env-first resolution the engine uses.

Run:  DSH_EDITAPART_E2E_SRC=/path/to/footage.mp4 <venv>/bin/python tests/test_loop_e2e.py -v
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
CORE = os.path.join(BIN, "edit_apart_core.py")
sys.path.insert(0, BIN)

SRC = os.environ.get("DSH_EDITAPART_E2E_SRC")
INVENTORY = os.environ.get("DSH_EDITAPART_E2E_INVENTORY")
PY = os.environ.get("DSH_EDIT_PY", sys.executable)


def brief(intent: str, pace: str, target: float, min_dur: float = 0.9) -> dict:
    return {"intent": intent, "pace": pace, "min_shot_dur": min_dur, "max_shot_dur": 6.0,
            "skip_short": True, "target_duration": target, "no_shot_under_s": min_dur}


def engine(*args) -> dict:
    r = subprocess.run([PY, CORE, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"engine {' '.join(args[:2])} failed: {r.stdout} {r.stderr}")
    return json.loads(r.stdout)


@unittest.skipUnless(SRC or INVENTORY,
                     "set DSH_EDITAPART_E2E_SRC (video) or DSH_EDITAPART_E2E_INVENTORY")
class TestLoopE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tastee2e_")
        if INVENTORY:
            with open(INVENTORY, encoding="utf-8") as fh:
                cls.inv = json.load(fh)
            cls.rendered = None
        else:
            cls.inv = engine("inventory", SRC)
            cls.inv = engine("features", SRC, json.dumps(cls.inv))
            with open(os.path.join(cls.tmp, "inventory.json"), "w", encoding="utf-8") as fh:
                json.dump(cls.inv, fh)
            cls.rendered = SRC
        # Both creators get the SAME brief family; only their revealed choice
        # among the rubric-equivalent candidates differs. That isolates taste
        # from the rubric.
        cls.corpus_briefs = [brief("editorial highlight", "medium", 14 + i * 0.8)
                             for i in range(25)]
        cls.neutral = [brief("editorial highlight", "medium", 15 + i * 0.9) for i in range(12)]

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _corpus(self, name: str, briefs: list[dict], longer: bool) -> str:
        """Drive the real loop: propose a group, log the creator's revealed pick
        (the candidate they would render) and the critic's dense reward for it."""
        ds = os.path.join(self.tmp, f"{name}.jsonl")
        for i, rub in enumerate(briefs):
            out = engine("propose", json.dumps(self.inv), json.dumps(rub),
                         "--group", "6", "--dataset", ds, "--clip-id", f"{name}{i}")
            cands = out["group"]["candidates"]
            picker = max if longer else min
            pick = picker(range(len(cands)), key=lambda j: cands[j]["duration"])
            # The revealed pick's OWN schema. Passing the whole propose output
            # would critique the SELECTED candidate's structure while labeling the
            # reward with --candidate pick, i.e. mislabel the field the trainer reads.
            crit = engine("critic", json.dumps(cands[pick]["schema"]), json.dumps(self.inv),
                          json.dumps(rub), "--dataset", ds,
                          "--group-id", out["group"]["group_id"],
                          "--candidate", str(pick), "--chosen-by", "creator")
            self.assertIn("logged", crit)
            # Regression guard: the logged reward must be the PICKED candidate's.
            # Passing the whole propose output (whose top-level structure is the
            # SELECTED candidate) critiques a different schema while labeling the
            # reward with --candidate pick.
            self.assertEqual(crit["reward"], cands[pick]["reward_obj"],
                             "the logged reward is not the picked candidate's")
            self.assertEqual(crit["logged"]["chosen_by"], "creator")
        return ds

    def test_loop_records_groups_and_revealed_picks(self):
        import taste_model as tm
        ds = self._corpus("probe", self.neutral[:3], longer=True)
        _, stats = tm.build_training_arrays(ds)
        self.assertEqual(stats["groups_usable"], 3)
        self.assertEqual(stats["reward_records"], 3)
        self.assertEqual(stats["bad_lines"], 0)
        groups, _ = tm.load_groups(ds)
        self.assertTrue(all(g["chosen_by"] == "creator" for g in groups))
        # the revealed pick must become the group's best reward
        prepared, _ = tm.build_training_arrays(ds)
        for p in prepared:
            self.assertEqual(int(p["rewards"].argmax()), p["chosen"],
                             "the revealed creator pick is not the top reward")

    def test_per_creator_identity_changes_the_selected_edit(self):
        import taste_model as tm

        ds_long = self._corpus("longtake", self.corpus_briefs, longer=True)
        ds_short = self._corpus("shorttake", self.corpus_briefs, longer=False)
        union = os.path.join(self.tmp, "union.jsonl")
        with open(union, "w", encoding="utf-8") as out:
            for p in (ds_long, ds_short):
                with open(p, encoding="utf-8") as fh:
                    out.write(fh.read())

        style = os.path.join(self.tmp, "style_brain_video_v1.gguf")
        general = os.path.join(self.tmp, "general.gguf")
        tm.train(union, general, style=style, creator="general", epochs=150, seed=0)
        self.assertTrue(os.path.exists(style))

        # Per-creator identities: ONLY z_u is trained, the trunk stays shared.
        ident = {}
        for name, ds in (("longtake", ds_long), ("shorttake", ds_short)):
            path = os.path.join(self.tmp, f"{name}.gguf")
            m = tm.train(ds, path, style=style, creator=name, epochs=300, seed=0,
                         freeze_style=True)
            self.assertTrue(m["style_frozen"])
            # A PERFECT fit is not the bar: the earlier 1.0 was measured under a
            # degenerate objective that only reinforced positive advantages (see
            # TestObjectiveFidelity). What matters is that the identity separates
            # its own corpus well above chance (1/K) before we ask about transfer.
            self.assertGreaterEqual(m["history"][-1]["train_rank_acc"], 0.6,
                                    f"{name} did not fit its own revealed preferences")
            ident[name] = path
        # the shared trunk must be byte-identical after both identity runs
        digests = {tm.TasteScorer.load(p, style=style).model.shared_digest()
                   for p in ident.values()}
        self.assertEqual(len(digests), 1, "per-creator runs changed the shared style-brain")

        picks = {"longtake": [], "shorttake": []}
        for rub in self.neutral:
            for name in ("longtake", "shorttake"):
                out = engine("propose", json.dumps(self.inv), json.dumps(rub),
                             "--group", "6", "--identity", ident[name],
                             "--style", style, "--select", "taste", "--no-log")
                g = out["group"]
                self.assertEqual(g["select_by"], "taste")
                scores = [c["taste_score"] for c in g["candidates"]]
                self.assertEqual(g["selected"], max(range(len(scores)), key=lambda i: scores[i]),
                                 "the selection is not the taste argmax")
                picks[name].append(g["candidates"][g["selected"]]["duration"])
            # determinism: same brief + same identity ⇒ same pick
            again = engine("propose", json.dumps(self.inv), json.dumps(rub),
                           "--group", "6", "--identity", ident["longtake"], "--style", style,
                           "--select", "taste", "--no-log")
            self.assertEqual(again["group"]["selected"],
                             engine("propose", json.dumps(self.inv), json.dumps(rub),
                                    "--group", "6", "--identity", ident["longtake"],
                                    "--style", style, "--select", "taste",
                                    "--no-log")["group"]["selected"])

        a, b = picks["longtake"], picks["shorttake"]
        mean_long = sum(a) / len(a)
        mean_short = sum(b) / len(b)
        differing = sum(1 for x, y in zip(a, b) if x != y)
        longer = sum(1 for x, y in zip(a, b) if x > y)
        print(f"\n  neutral briefs: {len(self.neutral)}")
        print(f"  mean selected duration  long-take={mean_long:.2f}s  short-take={mean_short:.2f}s")
        print(f"  briefs where the identities picked differently: {differing}/{len(self.neutral)}"
              f" (long-take longer in {longer})")
        print(f"  long-take picks:  {[round(x, 2) for x in a]}")
        print(f"  short-take picks: {[round(x, 2) for x in b]}")
        self.assertGreaterEqual(differing, len(self.neutral) // 2,
                                "the per-creator identity barely affected the loop's selection")
        self.assertGreater(mean_long, mean_short * 1.2,
                           "the long-take identity did not learn to select longer edits")

    def test_taste_selected_schema_still_renders_and_critiques(self):
        import taste_model as tm
        if not self.rendered:
            self.skipTest("no source video (inventory-only run)")
        ds = self._corpus("render", self.neutral[:2], longer=True)
        identity = os.path.join(self.tmp, "render.gguf")
        m = tm.train(ds, identity, creator="render", epochs=30, seed=0)
        style = m["style_brain"]
        out = engine("propose", json.dumps(self.inv), json.dumps(self.neutral[0]),
                     "--group", "4", "--identity", identity, "--style", style,
                     "--select", "taste", "--no-log")
        schema = json.dumps({"structure": out["structure"], "meta": out["meta"],
                             "globals": out["globals"]})
        mp4 = os.path.join(self.tmp, "taste_selected.mp4")
        rendered = engine("render", self.rendered, schema, mp4)
        self.assertTrue(os.path.exists(mp4))
        crit = engine("critic", schema, json.dumps(self.inv), json.dumps(self.neutral[0]))
        self.assertIn("overall_score", crit)
        revised = engine("revise", schema, json.dumps(crit))
        self.assertLessEqual(len(revised["structure"]), len(out["structure"]))
        print(f"\n  rendered taste-selected edit: {rendered.get('duration')}s, "
              f"critic overall {crit['overall_score']}, "
              f"{len(out['structure'])} -> {len(revised['structure'])} segments after revise")


class TestVideoRenderRobustness(unittest.TestCase):
    """Renders that must not fail, and malformed values that must fail clearly."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.tmp = tempfile.mkdtemp(prefix="renderrobust_")
        cls.mute = os.path.join(cls.tmp, "mute.mp4")
        cls.tone = os.path.join(cls.tmp, "tone.mp4")
        for path, extra in ((cls.mute, ["-an"]),
                            (cls.tone, ["-f", "lavfi", "-i", "sine=frequency=440", "-shortest"])):
            r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                "testsrc=size=320x240:rate=10", *extra, "-t", "2",
                                "-pix_fmt", "yuv420p", path],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise unittest.SkipTest(f"could not synthesize a test clip: {r.stderr[:200]}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def _schema(crop=None):
        tf = {"scale": 1.0, "crop": crop, "position": None}
        return {"structure": [{
            "shot": "clip_001", "trim": {"in": 0.0, "out": 1.0}, "retime": 0.0,
            "timeline": {"in": 0.0, "out": 1.0},
            "transition": {"type": "cut", "dur": 0.0, "params": {}},
            "transform": tf, "grade": {"lut": None, "eq": {}},
            "audio": {"level": 1.0, "duck": 0.0, "fade": 0.0}, "overlay": [], "why": "t"}]}

    def test_a_source_without_audio_still_renders(self):
        out = os.path.join(self.tmp, "mute_out.mp4")
        rendered = engine("render", self.mute, json.dumps(self._schema()), out)
        self.assertTrue(os.path.exists(out))
        # the silence fallback gives the output a real audio track
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                                "-show_entries", "stream=codec_type", "-of", "csv=p=0", out],
                               capture_output=True, text=True)
        self.assertTrue(probe.stdout.strip(), "expected a synthesized silent audio stream")
        self.assertIn("out", rendered)

    def test_a_source_with_audio_still_renders(self):
        out = os.path.join(self.tmp, "tone_out.mp4")
        engine("render", self.tone, json.dumps(self._schema()), out)
        self.assertTrue(os.path.exists(out))

    def test_selftest_reports_a_working_environment(self):
        """The video loop's conformance check: inventory + features + a real render."""
        if not (os.environ.get("DSH_SCENEDETECT") or shutil.which("scenedetect")):
            self.skipTest("scenedetect not on PATH and DSH_SCENEDETECT unset")
        out = engine("selftest")
        self.assertTrue(out["ok"])
        self.assertGreaterEqual(out["shots"], 1)
        self.assertGreater(out["mean_motion"], 0.0, "degenerate features")
        self.assertTrue(out["rendered_duration"])

    def test_a_hostile_crop_value_is_rejected_clearly(self):
        schema = self._schema(crop="800:600:0:0,movie=/etc/passwd")
        r = subprocess.run([PY, CORE, "render", self.tone, json.dumps(schema),
                            os.path.join(self.tmp, "nope.mp4")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("transform.crop", json.loads(r.stdout)["error"])

    def test_a_non_numeric_scale_is_rejected_clearly(self):
        schema = self._schema()
        schema["structure"][0]["transform"]["scale"] = "1.0,eq=brightness=1"
        r = subprocess.run([PY, CORE, "render", self.tone, json.dumps(schema),
                            os.path.join(self.tmp, "nope2.mp4")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("transform.scale", json.loads(r.stdout)["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
