#!/usr/bin/env python3
"""Experiment: is photo-taste transfer a DATA requirement or an OBJECTIVE failure?

Background: `docs/paper-findings.md` Finding 4c. A photo identity trained on ONE
image separates two creators with opposite framing taste 10/10 on held-out briefs
from the training family and 0/10 on briefs with an unseen crop geometry, because
crop geometry decides which pixels survive, so the crop-area feature is
confounded with the region statistics of the content it exposes.

Question: does varying the CONDITIONING across groups (more images, more crop
geometries) repair that, or does a static modality need a different objective?

DESIGN (v2 — revised after a read-only review killed v1)

v1 was uninterpretable: the arms were evaluated on their OWN training geometries,
the OOD cells held a single geometry whose eight "trials" were four distinct
inputs counted twice (n_eff = 4, so even 4/4 cannot reach significance), the
3-image arms received 2x the optimiser steps, and the stated chance rate (0.5) was
wrong (with no taste both identities pick the same candidate, so the direction
rate is 0). v2 fixes each:

  * ONE shared probe panel, identical for every arm, spanning a distance ladder
    away from the fixed family's manifold (in_family -> near -> mid -> far).
  * Both image conditions (a trained image, a never-trained image) x every probe.
  * Equal total optimiser steps across arms (epochs are derived per arm from its
    batch count, not fixed across arms).
  * A continuous metric: Kendall tau between an identity's taste scores and the
    candidates' crop areas, averaged over briefs. Chance is 0 for either identity;
    the estimand is the SEPARATION tau_loose - tau_tight.
  * Distinct brief inputs per cell (no duplicated trials), and an explicit n.
  * The training fit metrics and the per-creator `crop_area` weight are recorded
    rather than discarded (Finding 1b: a negative result is only as trustworthy
    as the model's demonstrated ability to express the thing at all).
  * A taste-irrelevant CONTROL identity (always picks the same candidate index) on
    the two extreme arms; its tau on crop_area must be ~0 for the metric to be
    trusted.

Arms (2x2):

    A1_1img_fixed   1 image,  narrow fixed crop family   (the Finding-4c setup)
    A2_1img_varied  1 image,  varied crop family
    B1_3img_fixed   3 images, narrow fixed crop family
    B2_3img_varied  3 images, varied crop family

Caveat this design does NOT remove, stated up front: the varied family's crop
areas overlap the lower probes, so a varied arm succeeding at "near"/"mid" may be
interpolating its own support rather than transferring. The ladder is what makes
that visible; it is not made to disappear.

Read the result as S(probe) per arm, with in_family as the manipulation check
(every arm that learned the taste should separate there) and far as the transfer
test. This experiment instantiates only the DATA half of the question: a null at
the far probe is consistent with the objective/architecture hypothesis but does
not establish it.

Run:  <venv>/bin/python tests/experiment_photo_transfer.py [--train-briefs 10] [--eval-briefs 3]
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
PHOTO_CORE = os.path.join(BIN, "photo_core.py")
sys.path.insert(0, BIN)

import taste_model as tm  # noqa: E402

PY = os.environ.get("DSH_EDIT_PY", sys.executable)
GROUP = 10          # grid entries 0..9 -> four distinct crop levels
TRAIN_IMAGES = 3
HELD_IMAGE = 3
TARGET_STEPS = 300  # equalised across arms

# Narrow family: mirrors the Finding-4c setup (area 0.248..0.275).
FIXED_FAMILY = [(0.05 + 0.01 * (i % 3), 0.08, 0.55 + 0.02 * (i % 4), 0.45)
                for i in range(12)]
# Varied family: written out because a truncated nested comprehension takes the
# values in the wrong nesting order (all tuples would share x0). Areas 0.144..0.461.
VARIED_FAMILY = [
    (0.02, 0.05, 0.40, 0.36), (0.16, 0.22, 0.58, 0.50), (0.30, 0.05, 0.72, 0.64),
    (0.02, 0.22, 0.72, 0.36), (0.16, 0.05, 0.40, 0.64), (0.30, 0.22, 0.58, 0.36),
    (0.02, 0.05, 0.58, 0.64), (0.16, 0.22, 0.72, 0.36), (0.30, 0.05, 0.40, 0.50),
    (0.02, 0.22, 0.40, 0.50), (0.16, 0.05, 0.72, 0.64), (0.30, 0.22, 0.72, 0.50),
]

# Shared probe panel: a distance ladder of crop AREAS away from the fixed family.
# in_family is in-support for both families; near is below the fixed family's
# minimum but inside the varied range; mid and far are below both.
PROBES = [
    ("in_family", (0.06, 0.08, 0.58, 0.46)),
    ("near", (0.12, 0.12, 0.42, 0.40)),
    ("mid", (0.20, 0.18, 0.34, 0.32)),
    ("far", (0.34, 0.30, 0.30, 0.26)),
]


def engine(*args) -> dict:
    env = dict(os.environ)
    for k in ("DSH_EDITAPART_DATASET", "DSH_EDITAPART_IDENTITY", "DSH_EDITAPART_STYLE",
              "DSH_EDITAPART_DATA"):
        env[k] = ""          # never inherit a workspace dataset/identity
    for attempt in (1, 2):
        r = subprocess.run([PY, PHOTO_CORE, *args], capture_output=True, text=True, env=env)
        if r.returncode == 0:
            return json.loads(r.stdout)
        print(f"  ! engine {args[0]} exit {r.returncode} (attempt {attempt}); "
              f"{r.stderr.strip()[:160]}", file=sys.stderr, flush=True)
    raise RuntimeError(f"photo engine {args[0]} failed twice: {r.stdout[:300]} {r.stderr[:300]}")


def make_image(path: str, cx: int, cy: int, hue: str) -> str:
    cmd = ["magick", "-size", "1600x1200", "xc:#232c38",
           "-fill", "#6b7684", "-draw", "rectangle 0,300 1600,560",
           "-fill", hue, "-draw", f"circle {cx},{cy} {cx},{cy - 180}",
           "-fill", "#0d1218", "-draw", "rectangle 100,860 700,1150",
           "-fill", "#8a5a2b", "-draw", "rectangle 900,780 1450,1160",
           "-fill", "#e8e8e8", "-draw", f"rectangle {cx - 120},{cy + 200} {cx + 120},{cy + 300}",
           path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"image generation failed: {r.stderr[:300]}")
    return path


def crop_area(operations: list[dict], inspect: dict) -> float:
    w = float(inspect.get("width") or 1.0)
    h = float(inspect.get("height") or 1.0)
    for op in operations:
        if op.get("op") == "crop":
            return (float(op["w"]) * float(op["h"])) / (w * h)
    return 1.0


def brief(geometry, target_luma: float) -> dict:
    return {"intent": "warm punch", "saturation": 1.15, "crop": {"percent": list(geometry)},
            "width": 900, "target_luma": target_luma}


def kendall_tau(a: list[float], b: list[float]) -> float:
    """tau over untied pairs; 0.0 when there is nothing untied."""
    conc = disc = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    n = conc + disc
    return (conc - disc) / n if n else 0.0


def build_corpus(images, geometries, briefs, picker, workdir, name) -> str:
    """One creator's log: propose the group once per (image, brief), then record
    that creator's revealed pick for the group."""
    ds = os.path.join(workdir, f"{name}.jsonl")
    corpus_dir = os.path.join(workdir, f"r_{name}")
    for img_i, (path, insp) in enumerate(images):
        for b in range(briefs):
            rub = brief(geometries[b % len(geometries)], 0.42 + 0.02 * (b % 6))
            out = engine("propose", path, json.dumps(insp), json.dumps(rub),
                         "--group", str(GROUP), "--dataset", ds,
                         "--clip-id", f"{name}_im{img_i}b{b}",
                         "--workdir", corpus_dir)
            cands = out["group"]["candidates"]
            pick = picker(cands, insp)
            engine("critic", json.dumps({"operations": cands[pick]["operations"]}),
                   json.dumps(rub), cands[pick]["render"], "--dataset", ds,
                   "--group-id", out["group"]["group_id"], "--candidate", str(pick),
                   "--chosen-by", "creator")
    return ds


def pick_area(extreme: str):
    def picker(cands, insp):
        areas = [crop_area(c["operations"], insp) for c in cands]
        return (min if extreme == "tight" else max)(range(len(areas)), key=lambda i: areas[i])
    return picker


def pick_index_fixed(idx: int):
    """Control creator: always chooses the same candidate INDEX, so it carries no
    crop-area rule. Its tau on crop_area must be ~0 for the metric to be trusted."""
    def picker(cands, insp):
        return min(idx, len(cands) - 1)
    return picker


def equalising_epochs(dataset: str, target_steps: int, batch_groups: int = 16) -> int:
    prepared, _ = tm.build_training_arrays(dataset)
    n = max(1, len(prepared))
    batches = max(1, -(-n // batch_groups))
    return max(1, target_steps // batches)


def train_arm(name, images, geometries, workdir):
    """Both creators' corpora, one shared trunk, two frozen identities trained to
    matched optimiser step counts. Returns artifacts + fit evidence."""
    n_briefs = len(geometries)
    ds_tight = build_corpus(images, geometries, n_briefs, pick_area("tight"),
                            os.path.join(workdir, f"{name}_tight"), f"{name}_tight")
    ds_loose = build_corpus(images, geometries, n_briefs, pick_area("loose"),
                            os.path.join(workdir, f"{name}_loose"), f"{name}_loose")
    union = os.path.join(workdir, f"{name}_union.jsonl")
    with open(union, "w", encoding="utf-8") as out:
        for p in (ds_tight, ds_loose):
            with open(p, encoding="utf-8") as fh:
                out.write(fh.read())

    style = os.path.join(workdir, f"{name}_style_brain_photo_v1.gguf")
    trunk = tm.train(union, os.path.join(workdir, f"{name}_general.gguf"), style=style,
                     creator="general", epochs=150, seed=0)

    out = {"style": style, "ident": {}, "fit": {}, "epochs": {},
           "trunk_groups": trunk["train_groups"]}
    for who, ds in (("tight", ds_tight), ("loose", ds_loose)):
        epochs = equalising_epochs(ds, TARGET_STEPS)
        out["epochs"][who] = epochs
        path = os.path.join(workdir, f"{name}_{who}.gguf")
        m = tm.train(ds, path, style=style, creator=who, epochs=epochs, seed=0,
                     freeze_style=True)
        out["ident"][who] = path
        out["fit"][who] = {
            "epochs": epochs,
            "train_groups": m["train_groups"],
            "train_rank_acc": round(float(m["history"][-1]["train_rank_acc"]), 3),
            "train_pair_agree": round(float(m["history"][-1]["train_pair_agree"]), 3),
            "eval_pair_agree": round(float(m["eval_pair_agree_after"]), 3),
            "z_u_norm": round(float(m["z_u_norm"]), 3),
            # effective optimiser steps, so the step-matching claim is checkable
            "steps": epochs * max(1, -(-int(m["train_groups"]) // 16)),
        }
    j = tm.PHOTO_FEATURES.index("crop_area")
    for who, path in out["ident"].items():
        scorer = tm.TasteScorer.load(path, style=style)
        out["fit"][who]["weight_crop_area"] = round(
            float((scorer.model.p["z_u"] @ scorer.model.p["Wl"])[j]), 4)
    return out


def eval_cell(images, geometry, eval_briefs, arm, workdir, tag):
    """One (image-condition, probe) cell: per identity, the Kendall tau between its
    taste scores and the candidates' crop areas."""
    per = {"tight": {"taus": [], "picks": []}, "loose": {"taus": [], "picks": []}}
    for k in range(eval_briefs):
        path, insp = images[k % len(images)]
        rub = brief(geometry, 0.44 + 0.04 * k)      # distinct inputs: no fake n
        for who in ("tight", "loose"):
            out = engine("propose", path, json.dumps(insp), json.dumps(rub),
                         "--group", str(GROUP), "--identity", arm["ident"][who],
                         "--style", arm["style"], "--select", "taste", "--no-log",
                         "--workdir", os.path.join(workdir, f"e_{tag}"))
            cands = out["group"]["candidates"]
            areas = [crop_area(c["operations"], insp) for c in cands]
            scores = [c["taste_score"] for c in cands]
            per[who]["taus"].append(kendall_tau(scores, areas))
            per[who]["picks"].append(areas[out["group"]["selected"]])
    return per


def summarise(per, eval_briefs):
    tau_t = sum(per["tight"]["taus"]) / len(per["tight"]["taus"])
    tau_l = sum(per["loose"]["taus"]) / len(per["loose"]["taus"])
    return {
        "tau_tight": round(tau_t, 4), "tau_loose": round(tau_l, 4),
        "separation": round(tau_l - tau_t, 4),
        "mean_pick_tight": round(sum(per["tight"]["picks"]) / len(per["tight"]["picks"]), 4),
        "mean_pick_loose": round(sum(per["loose"]["picks"]) / len(per["loose"]["picks"]), 4),
        # Behavioural effect: what the loop actually SELECTS (ratio < 1 = the tight
        # identity framed tighter). tau above measures the learned ORDERING; this
        # measures the DECISION. They can disagree, and when they do the decision
        # is what the user sees.
        "pick_ratio": round(
            (sum(per["tight"]["picks"]) / len(per["tight"]["picks"])) /
            max(1e-9, sum(per["loose"]["picks"]) / len(per["loose"]["picks"])), 4),
        "picked_differently": sum(1 for x, y in zip(per["tight"]["picks"],
                                                    per["loose"]["picks"]) if x != y),
        "tight_tighter": sum(1 for x, y in zip(per["tight"]["picks"],
                                               per["loose"]["picks"]) if x < y),
        "n_distinct_inputs": eval_briefs,
    }


def control_tau(arm, cpath, images, eval_briefs, workdir, name):
    """The control identity's tau on crop_area, on the two decisive probes; ~0
    means the metric is not just reporting arbitrary reordering."""
    out = {}
    for probe, geom in (p for p in PROBES if p[0] in ("in_family", "far")):
        taus = []
        for k in range(eval_briefs):
            path, insp = images[0]
            rub = brief(geom, 0.44 + 0.04 * k)
            res = engine("propose", path, json.dumps(insp), json.dumps(rub),
                         "--group", str(GROUP), "--identity", cpath,
                         "--style", arm["style"], "--select", "taste", "--no-log",
                         "--workdir", os.path.join(workdir, f"ctrl_{name}_{probe}"))
            cands = res["group"]["candidates"]
            areas = [crop_area(c["operations"], insp) for c in cands]
            scores = [c["taste_score"] for c in cands]
            taus.append(kendall_tau(scores, areas))
        out[probe] = round(sum(taus) / len(taus), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-briefs", type=int, default=10)
    ap.add_argument("--eval-briefs", type=int, default=3)
    ap.add_argument("--no-control", dest="control", action="store_false", default=True,
                    help="skip the taste-irrelevant control identity")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default=None,
                    help="parent dir for the run (default ~/.dsh/work/photo-transfer)")
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    if shutil.which("magick") is None and shutil.which("convert") is None:
        print("ImageMagick is required for this experiment", file=sys.stderr)
        return 2

    # Candidate renders are PNGs of a 1600x1200 source; /tmp here is a 31 GiB
    # TMPFS, so accumulating thousands of them consumes RAM. Keep them on the real
    # filesystem instead.
    base = args.workdir or os.path.join(os.path.expanduser("~"), ".dsh", "work",
                                        "photo-transfer")
    os.makedirs(base, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="run_", dir=base)
    print(f"workdir: {workdir}", flush=True)
    try:
        specs = [(1150, 420, "#ffe9a8"), (400, 320, "#cfe8ff"),
                 (980, 980, "#ffd0b0"), (1300, 760, "#d8ffd0")]
        paths = [make_image(os.path.join(workdir, f"im{i}.png"), *s)
                 for i, s in enumerate(specs)]
        images = [(p, engine("inspect", p)) for p in paths]

        fixed = FIXED_FAMILY[:args.train_briefs]
        varied = VARIED_FAMILY[:args.train_briefs]
        arms = [
            ("A1_1img_fixed", images[:1], fixed),
            ("A2_1img_varied", images[:1], varied),
            ("B1_3img_fixed", images[:TRAIN_IMAGES], fixed),
            ("B2_3img_varied", images[:TRAIN_IMAGES], varied),
        ]
        fam_min = {k: min(round(g[2] * g[3], 4) for g in fam)
                   for k, fam in (("fixed", FIXED_FAMILY), ("varied", VARIED_FAMILY))}
        fam_max = {k: max(round(g[2] * g[3], 4) for g in fam)
                   for k, fam in (("fixed", FIXED_FAMILY), ("varied", VARIED_FAMILY))}
        print(f"fixed-family crop areas:  {fam_min['fixed']} .. {fam_max['fixed']}", flush=True)
        print(f"varied-family crop areas: {fam_min['varied']} .. {fam_max['varied']}", flush=True)
        for label, g in PROBES:
            a = round(g[2] * g[3], 4)
            where = ("inside both" if a >= fam_min["fixed"] else
                     "inside varied only" if a >= fam_min["varied"] else
                     "below both (extrapolation)")
            print(f"  probe {label:<9} area {a:<7} -> {where}", flush=True)

        results = {}
        for name, imgs, geoms in arms:
            print(f"\n=== arm {name}: {len(imgs)} image(s) x {len(geoms)} briefs ===", flush=True)
            arm = train_arm(name, imgs, geoms, workdir)
            arm["images"] = len(imgs)
            arm["geometry"] = "fixed" if geoms is fixed else "varied"
            arm["cells"] = {}
            for cond_name, eval_imgs in (("seen_image", [images[0]]),
                                         ("unseen_image", [images[HELD_IMAGE]])):
                for probe, geom in PROBES:
                    per = eval_cell(eval_imgs, geom, args.eval_briefs, arm, workdir,
                                    f"{name}_{cond_name}_{probe}")
                    arm["cells"][f"{cond_name}/{probe}"] = summarise(per, args.eval_briefs)
            if args.control and name in ("A1_1img_fixed", "B2_3img_varied"):
                ctrl = build_corpus(imgs, geoms, min(8, len(geoms)), pick_index_fixed(5),
                                    os.path.join(workdir, f"{name}_ctrl"), f"{name}_ctrl")
                cpath = os.path.join(workdir, f"{name}_ctrl.gguf")
                cm = tm.train(ctrl, cpath, style=arm["style"], creator="control",
                              epochs=equalising_epochs(ctrl, TARGET_STEPS), seed=0,
                              freeze_style=True)
                arm["control_fit"] = {
                    "train_rank_acc": round(float(cm["history"][-1]["train_rank_acc"]), 3),
                    "z_u_norm": round(float(cm["z_u_norm"]), 3),
                }
                arm["control_tau"] = control_tau(arm, cpath, imgs, args.eval_briefs,
                                                workdir, name)
            results[name] = arm
            print(json.dumps({"fit": arm["fit"], "epochs": arm["epochs"],
                              "trunk_groups": arm["trunk_groups"]}, indent=2), flush=True)

        print("\n=== SEPARATION  tau_loose - tau_tight   (chance 0.00; in_family is the "
              "manipulation check, far is the transfer test) ===")
        print(f"{'arm':<16} {'fit t/l':<10} {'img':<4} " +
              " ".join(f"{p[:9]:>9}" for p, _ in PROBES))
        for name, arm in results.items():
            fit = (f"{arm['fit']['tight']['train_rank_acc']:.2f}/"
                   f"{arm['fit']['loose']['train_rank_acc']:.2f}")
            for cond in ("seen_image", "unseen_image"):
                row = " ".join(f"{arm['cells'][f'{cond}/{p}']['separation']:>9.3f}"
                               for p, _ in PROBES)
                imgcol = str(arm["images"]) if cond == "seen_image" else ""
                print(f"{name if cond == 'seen_image' else '':<16} "
                      f"{fit if cond == 'seen_image' else '':<10} {imgcol:<4} {row}"
                      f"   [{cond}]")
        print("\n=== BEHAVIOURAL  mean selected crop area, tight/loose ratio  "
              "(1.00 = no effect; < 1 = tight framed tighter) ===")
        print(f"{'arm':<16} " + " ".join(f"{p[:9]:>9}" for p, _ in PROBES))
        for name, arm in results.items():
            for cond in ("seen_image", "unseen_image"):
                row = " ".join(f"{arm['cells'][f'{cond}/{p}']['pick_ratio']:>9.3f}"
                               for p, _ in PROBES)
                print(f"{name if cond == 'seen_image' else '':<16} {row}   [{cond}]")

        print("\nper-creator weight on crop_area (negative = prefers tight):")
        for name, arm in results.items():
            print(f"  {name:<16} tight {arm['fit']['tight']['weight_crop_area']:+.4f}  "
                  f"loose {arm['fit']['loose']['weight_crop_area']:+.4f}")
        print("\ncontrol identity (same candidate index always) tau on crop_area, must be ~0:")
        for name, arm in results.items():
            if "control_tau" in arm:
                print(f"  {name}: {arm['control_tau']}  fit={arm['control_fit']}")
        print(f"\nn_distinct_inputs per cell = {args.eval_briefs}; step target = {TARGET_STEPS}; "
              "epochs: " + ", ".join(f"{n}={a['epochs']['tight']}" for n, a in results.items()))
        print("paired-pick counts live in each cell's 'tight_tighter'/'picked_differently'.")

        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, sort_keys=True, default=str)
            print(f"\nwrote {args.out}")
        return 0
    finally:
        if args.keep_workdir:
            print(f"kept workdir: {workdir}", flush=True)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
