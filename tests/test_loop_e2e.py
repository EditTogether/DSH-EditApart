#!/usr/bin/env python3
"""End-to-end test of the wired-in taste loop on REAL footage.

This is the test that answers "is the identity learning actually part of the
editor, or still a side program?":

  1. build an enriched shot inventory from a real video (scenedetect + ffmpeg),
  2. drive the real loop over two creators who reveal different tastes: when a
     group of rubric-equivalent candidates is proposed, the "long-take" creator
     renders the longest candidate and the "short-take" creator the shortest.
     The critic's dense reward is logged for the rendered pick, and the pick is
     logged with `chosen_by="creator"` so it enters training as a REVEALED
     preference (without that flag the rubric-driven objective reward dominates
     and both creators converge to the same edits),
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
            crit = engine("critic", json.dumps(out), json.dumps(self.inv),
                          json.dumps(rub), "--dataset", ds,
                          "--group-id", out["group"]["group_id"],
                          "--candidate", str(pick), "--chosen-by", "creator")
            self.assertIn("logged", crit)
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
            self.assertEqual(m["history"][-1]["train_rank_acc"], 1.0,
                             f"{name} did not even fit its own revealed preferences")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
