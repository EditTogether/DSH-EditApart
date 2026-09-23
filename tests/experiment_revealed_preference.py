#!/usr/bin/env python3
"""Experiment: the Finding-1 ablation — must the user's revealed choice be a term
in the objective, and does *how* it enters matter?

`docs/paper-findings.md` Finding 1 claims per-user taste is unidentifiable from a
rubric-derived reward, and that the fix is to put the user's revealed choice into
the objective. Finding 1b adds that the earlier, opposite conclusion ("only
overriding the reward works") was an artifact of an adapter that could not express
the preference.

This script re-runs the whole ablation from scratch on the CURRENT code:

  A   no user-dependent term in the objective         (bonus=0, pref=0)
  B1  preference loss at the default weight           (bonus=0, pref=0.5)
  B2  preference loss at 4x                           (bonus=0, pref=2.0)
  C1  revealed-preference reward override only        (bonus=1, pref=0)
  C2  override + preference loss                      (bonus=1, pref=0.5)

Two creators with opposite taste (longest vs shortest take) are trained on the
same candidate groups with one shared frozen trunk per arm; the estimand is how
often, on held-out neutral briefs, the two identities pick a different edit — and
in the predicted direction.

NOTE: every earlier version of this table was measured before the GRPO ratio
baseline was corrected (a review found it used raw logits, so `rho = 1/Z` and
negative advantages lost their direct gradient). Numbers from that era are NOT
comparable and are superseded by this script's output.

Run:  <venv>/bin/python tests/experiment_revealed_preference.py [--briefs 25] [--eval 12]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "bin")
CORE = os.path.join(BIN, "edit_apart_core.py")
sys.path.insert(0, BIN)

import taste_model as tm  # noqa: E402

PY = os.environ.get("DSH_EDIT_PY", sys.executable)
GROUP = 6
TARGET_STEPS = 300


def engine(*args) -> dict:
    env = dict(os.environ)
    for k in ("DSH_EDITAPART_DATASET", "DSH_EDITAPART_IDENTITY", "DSH_EDITAPART_STYLE",
              "DSH_EDITAPART_DATA"):
        env[k] = ""
    for attempt in (1, 2):
        r = subprocess.run([PY, CORE, *args], capture_output=True, text=True, env=env)
        if r.returncode == 0:
            return json.loads(r.stdout)
        print(f"  ! engine {args[0]} exit {r.returncode}: {r.stderr.strip()[:120]}",
              file=sys.stderr, flush=True)
    raise RuntimeError(f"engine {args[0]} failed twice: {r.stdout[:200]}")


def brief(target: float, pace: str = "medium", min_dur: float = 0.9) -> dict:
    return {"intent": "editorial highlight", "pace": pace, "min_shot_dur": min_dur,
            "max_shot_dur": 6.0, "skip_short": True, "target_duration": target,
            "no_shot_under_s": min_dur}


def build_corpus(inventory, briefs, picker, workdir, name) -> str:
    """One creator's log: propose a group per brief, then log the creator's pick."""
    ds = os.path.join(workdir, f"{name}.jsonl")
    for i, rub in enumerate(briefs):
        # The VIDEO propose needs no workdir: its objective critic is arithmetic on
        # the schema + inventory, so a group costs no renders (unlike photo).
        out = engine("propose", json.dumps(inventory), json.dumps(rub), "--group", str(GROUP),
                     "--dataset", ds, "--clip-id", f"{name}{i}")
        cands = out["group"]["candidates"]
        pick = picker(range(len(cands)), key=lambda j: cands[j]["duration"])
        engine("critic", json.dumps(cands[pick]["schema"]), json.dumps(inventory),
               json.dumps(rub), "--dataset", ds, "--group-id", out["group"]["group_id"],
               "--candidate", str(pick), "--chosen-by", "creator")
    return ds


def equalising_epochs(dataset: str, target_steps: int, batch_groups: int = 16) -> int:
    prepared, _ = tm.build_training_arrays(dataset)
    batches = max(1, -(-max(1, len(prepared)) // batch_groups))
    return max(1, target_steps // batches)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=os.path.expanduser(
        "~/.dsh/work/avedit/core_invf.json"))
    ap.add_argument("--briefs", type=int, default=25)
    ap.add_argument("--eval", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.inventory, encoding="utf-8") as fh:
        inventory = json.load(fh)

    workdir = tempfile.mkdtemp(prefix="revealedpref_", dir=os.path.join(
        os.path.expanduser("~"), ".dsh", "work"))
    print(f"workdir: {workdir}", flush=True)
    try:
        train_briefs = [brief(14 + 0.8 * i) for i in range(args.briefs)]
        neutral = [brief(15 + 0.9 * i) for i in range(args.eval)]

        corpora = {}
        for name, picker in (("longtake", max), ("shorttake", min)):
            corpora[name] = build_corpus(inventory, train_briefs, picker, workdir, name)
            print(f"  built {name}: {args.briefs} groups", flush=True)
        union = os.path.join(workdir, "union.jsonl")
        with open(union, "w", encoding="utf-8") as out:
            for p in corpora.values():
                with open(p, encoding="utf-8") as fh:
                    out.write(fh.read())

        arms = [("A_no_user_term", 0.0, 0.0), ("B1_pref_loss", 0.0, 0.5),
                ("B2_pref_loss_x4", 0.0, 2.0), ("C1_override_only", 1.0, 0.0),
                ("C2_override_plus_loss", 1.0, 0.5)]
        results = {}
        for label, bonus, pref in arms:
            style = os.path.join(workdir, f"{label}_style.gguf")
            tm.train(union, os.path.join(workdir, f"{label}_general.gguf"), style=style,
                     creator="general", epochs=150, seed=0)
            ident, fit = {}, {}
            for who, ds in corpora.items():
                p = os.path.join(workdir, f"{label}_{who}.gguf")
                m = tm.train(ds, p, style=style, creator=who, epochs=equalising_epochs(ds, TARGET_STEPS),
                             seed=0, freeze_style=True, revealed_pref_bonus=bonus, pref_coef=pref)
                ident[who] = p
                fit[who] = round(float(m["history"][-1]["train_rank_acc"]), 3)
            durations = {"longtake": [], "shorttake": []}
            for k, rub in enumerate(neutral):
                for who in ident:
                    out = engine("propose", json.dumps(inventory), json.dumps(rub),
                                 "--group", str(GROUP), "--identity", ident[who],
                                 "--style", style, "--select", "taste", "--no-log")
                    g = out["group"]
                    durations[who].append(g["candidates"][g["selected"]]["duration"])
            a, b = durations["longtake"], durations["shorttake"]
            differing = sum(1 for x, y in zip(a, b) if x != y)
            longer = sum(1 for x, y in zip(a, b) if x > y)
            results[label] = {
                "bonus": bonus, "pref_coef": pref, "fit_train_rank_acc": fit,
                "differing": differing, "in_predicted_direction": longer, "n": len(a),
                "mean_longtake": round(sum(a) / len(a), 2),
                "mean_shorttake": round(sum(b) / len(b), 2),
            }
            print(f"  {label}: {differing}/{len(a)} differ, {longer}/{len(a)} in direction, "
                  f"{results[label]['mean_longtake']}s vs {results[label]['mean_shorttake']}s",
                  flush=True)

        print("\n=== Finding 1 ablation (current objective) ===")
        print(f"{'arm':<22} {'differ':>8} {'direction':>10} {'longtake':>9} {'shorttake':>10} "
              f"{'fit(l/s)':>12}")
        for label, r in results.items():
            print(f"{label:<22} {r['differing']:>4}/{r['n']:<3} "
                  f"{r['in_predicted_direction']:>6}/{r['n']:<3} "
                  f"{r['mean_longtake']:>8.2f}s {r['mean_shorttake']:>9.2f}s "
                  f"{str(r['fit_train_rank_acc']):>12}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, sort_keys=True)
            print(f"\nwrote {args.out}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
