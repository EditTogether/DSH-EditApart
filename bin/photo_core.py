#!/usr/bin/env python3
"""Photo (single-image) edit engine for the AI Taste Video Editor preset.

This is the image-only analogue of `edit_apart_core.py`. It is deliberately
import-free (stdlib + subprocess) and deterministic, and it needs ONE fewer
input modality than the video core: photo edits read a single IMAGE, so the
general model only needs the `image` modality — no `video` modality and no
video-in/out registration. ImageMagick (`magick`) is the deterministic renderer.

Subcommands (JSON on stdout, {error} + exit 2 on failure):
    inspect SRC                    -> per-image features + a spatial region grid
    propose SRC INSPECT RUBRIC [--group K --identity ID --dataset LOG ...]
                                   -> photo EDIT SCHEMA, or a GROUP of candidates
    render SRC SCHEMA OUT          -> deterministic magick render
    critic SCHEMA RUBRIC RESULT [--dataset LOG --group-id G --candidate I --chosen-by WHO]
                                   -> score + per-op deltas (+ optional reward logging)
    revise SCHEMA CRITIC           -> drop flagged operations
    taste_train / identity_init / taste_status / taste
                                   -> the shared per-creator model (bin/taste_model.py)
                                      with photo/v1 defaults, so the photo loop is
                                      self-contained

Taste wiring (same model as the video loop — see bin/taste_model.py, spec
`photo/v1`). A photo edit has no temporal axis, so this loop differs from video
in three ways the callers must know about:

  * Candidate scoring needs a RENDER. `critic` scores the rendered image, so a
    group of K candidates is rendered and scored (K magick runs) — unlike video,
    whose objective critic is arithmetic on the schema + inventory. With
    `--group 1` and no dataset nothing is rendered and the output is exactly the
    legacy single-schema proposal.
  * The feature layout is spatial: crop geometry plus the statistics of the
    region grid the crop keeps (`inspect`'s `regions`). There is no beat, no
    pacing, no shot list to lean on.
  * The objective reward is COARSE and partly quantised (`exposure_ok` is a
    tolerance). It used to be a hard threshold, which tied whole candidate
    groups and left the group-relative advantage identically zero; the score is
    now graded (full credit inside the tolerance exactly as before, linear
    falloff outside). Ties can still happen, so each logged group records `tie`
    and `reward_spread`, and `taste_status` reports `zero_spread_groups` — check
    that before concluding anything about a training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

# The renderer is resolved at runtime, NOT hard-locked to one path: a deployment
# may point DSH_IMAGEMAGICK at a managed binary, or lean on PATH, or (web profile)
# supply an in-browser ImageMagick/ffmpeg through a store plugin instead. This is
# deliberately not a hard dependency — the tool reports a clear error if no
# renderer is reachable rather than assuming /usr/bin/magick exists.
def _resolve_magick() -> str:
    env = os.getenv("DSH_IMAGEMAGICK")
    if env:
        return env
    found = shutil.which("magick") or shutil.which("convert")
    return found or "/usr/bin/magick"


MAGICK = _resolve_magick()


def _taste():
    """Import bin/taste_model.py (sitting next to this file), lazily, so the
    legacy inspect/propose/render/critic/revise path never needs numpy."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import taste_model  # noqa: PLC0415 — deliberate lazy import
    return taste_model


def _log_jsonl(path: str, record: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _fixture(value: str) -> dict:
    if isinstance(value, dict):
        return value
    if os.path.exists(value):
        with open(value, encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(value)


def run(cmd: list[str], binary: bool = False) -> subprocess.CompletedProcess[bytes | str]:
    return subprocess.run(cmd, capture_output=True, text=not binary)


def _ensure_magick() -> None:
    if shutil.which(MAGICK) is None and not os.path.exists(MAGICK):
        raise RuntimeError(
            f"no ImageMagick renderer reachable (tried {MAGICK}); set DSH_IMAGEMAGICK, "
            "install magick, or use an in-browser ImageMagick renderer plugin for the web profile",
        )


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------
def _dims(src: str) -> list[int]:
    r = run([MAGICK, "identify", "-format", "%w %h", src])
    if r.returncode != 0 or not r.stdout.strip():
        return [0, 0]
    try:
        return [int(v) for v in r.stdout.strip().split()[:2]]
    except ValueError:
        return [0, 0]


def _luma_samples(src: str) -> bytes:
    """Downscale to gray raw bytes for a cheap luminance histogram."""
    r = run([MAGICK, src, "-colorspace", "Gray", "-resize", "64x64!",
             "-depth", "8", "gray:-"], binary=True)
    if r.returncode != 0 or not r.stdout:
        return b""
    return r.stdout


def _rgb_stats(src: str, sample: int = 16, grid: int = 4
               ) -> tuple[list[float], float, list[list[dict]]]:
    """Mean RGB, mean saturation, and a `grid` x `grid` spatial map.

    One raw RGB downscale serves all three: the global numbers stay exactly what
    they were (so the documented reference run does not move), and the grid is a
    block-average of that same read. The grid is what gives a photo edit any
    spatial structure at all — it lets a crop's features describe WHERE the crop
    lands and WHAT the kept region looks like, which is the job the shot
    inventory does for video.
    """
    r = run([MAGICK, src, "-resize", f"{sample}x{sample}!", "-depth", "8", "rgb:-"], binary=True)
    data = r.stdout if r.returncode == 0 else b""
    if not data:
        return [0.0, 0.0, 0.0], 0.0, []
    px = [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data) - 2, 3)]
    n = len(px) or 1
    mean = [sum(p[c] for p in px) / n / 255.0 for c in range(3)]
    sat = sum((max(p) - min(p)) / 255.0 for p in px) / n
    cells: list[list[dict]] = []
    if grid > 0 and sample % grid == 0 and len(data) >= sample * sample * 3:
        block = sample // grid
        for row in range(grid):
            line: list[dict] = []
            for col in range(grid):
                acc_l = acc_s = 0.0
                for y in range(row * block, (row + 1) * block):
                    for x in range(col * block, (col + 1) * block):
                        i = (y * sample + x) * 3
                        pr, pg, pb = data[i], data[i + 1], data[i + 2]
                        acc_l += (0.299 * pr + 0.587 * pg + 0.114 * pb) / 255.0
                        acc_s += (max(pr, pg, pb) - min(pr, pg, pb)) / 255.0
                cnt = block * block
                line.append({"luma": round(acc_l / cnt, 3), "sat": round(acc_s / cnt, 3)})
            cells.append(line)
    return [round(v, 3) for v in mean], round(sat, 3), cells


def _percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    return sv[min(len(sv) - 1, int(round(p * (len(sv) - 1))))]


def cmd_inspect(src: str) -> dict:
    _ensure_magick()
    dims = _dims(src)
    lumas = list(_luma_samples(src))
    mean_rgb, sat_mean, cells = _rgb_stats(src)
    mean_luma = round(sum(lumas) / len(lumas) / 255.0, 3) if lumas else 0.0
    std = (sum((x / 255.0 - mean_luma) ** 2 for x in lumas) / len(lumas)) ** 0.5 if lumas else 0.0
    return {
        "source": src,
        "width": dims[0], "height": dims[1],
        "mean_luma": round(mean_luma, 3),
        "luma_std": round(std, 3),
        "lum_p50": round(_percentile(lumas, 0.5) / 255.0, 3),
        "lum_p95": round(_percentile(lumas, 0.95) / 255.0, 3),
        "saturation": sat_mean,
        "mean_rgb": mean_rgb,
        # Spatial structure. `regions.cells` is row-major from the top-left; a
        # crop rect maps to cells by fraction of the frame, so the taste features
        # can describe composition without rendering anything.
        "regions": {"grid": len(cells), "cells": cells},
    }


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------
# Deterministic candidate grid: (grade factor, crop-area factor, sharpen factor,
# caption-size factor). Candidate 0 is the rubric verbatim, so `group=1` returns
# exactly the legacy schema. The spread is deliberately wide: the objective
# photo reward is quantised (a threshold on mean luma), so near-duplicate
# candidates tie and a narrow grid yields no group-relative signal at all.
# (grade factor, crop-area factor, sharpen factor, caption-size factor,
#  exposure nudge, contrast nudge). The nudges are additive and are applied ONLY
# when the rubric is silent on that axis, so an explicit instruction is never
# overridden. The first five entries deliberately probe the axes the objective
# reward can SEE (rendered luma and luma spread); the rest probe taste axes the
# reward is blind to (crop geometry, sharpening, caption size), which the
# revealed preference has to carry.
# ORDER MATTERS: a small group takes a prefix, so the entries interleave the
# reward-visible axes (rendered luma, luma spread) with the taste axes the reward
# is blind to (crop geometry, sharpening, caption size). A 4-candidate group
# therefore still spans legacy / brighter / tighter / looser.
PHOTO_CANDIDATE_GRID = [
    (1.0, 1.0, 1.0, 1.0, 0.0, 0.0),      # legacy: rubric verbatim
    (1.0, 1.0, 1.0, 1.0, 0.15, 0.0),     # brighter (moves the scored luma)
    (1.0, 0.6, 1.0, 1.0, 0.0, 0.0),      # tighter crop
    (1.0, 1.5, 1.0, 1.0, 0.0, 0.0),      # looser crop
    (1.0, 1.0, 1.0, 1.0, -0.10, 0.0),    # darker
    (1.0, 1.0, 1.0, 1.0, 0.0, 0.35),     # punchier contrast
    (0.5, 1.0, 1.0, 1.0, 0.0, 0.0),      # muted grade
    (1.5, 1.0, 1.0, 1.0, 0.0, 0.0),      # strong grade
    (1.0, 0.4, 1.0, 1.0, 0.0, 0.0),      # much tighter crop
    (1.0, 1.8, 1.0, 1.0, 0.0, 0.0),      # much looser crop
    (1.0, 1.0, 2.0, 1.0, 0.0, 0.0),      # sharper
    (1.5, 0.6, 1.5, 1.0, 0.0, 0.0),      # strong + tight + sharp
    (0.5, 1.5, 0.5, 1.2, 0.0, 0.0),      # muted + loose + soft
    (1.0, 1.3, 1.0, 1.0, 0.15, 0.0),     # brighter + loose
]


def _legacy_ops(src: str, inspect: dict, rubric: dict) -> list[dict]:
    """The original rubric -> operations mapping. Unchanged: this is candidate 0."""
    intent = rubric.get("intent", "")
    grade: dict = {}
    sat_target = rubric.get("saturation", 1.0)
    if sat_target != 1.0:
        grade["saturation"] = sat_target
    exposure = rubric.get("exposure", 1.0)
    if exposure != 1.0:
        grade["exposure"] = exposure
    ops: list[dict] = []
    crop = rubric.get("crop")
    if crop:
        ops.append({"op": "crop", **_fraction_crop(crop, inspect), "why": "composition crop from rubric"})
    if grade:
        ops.append({"op": "grade", **_full_grade(grade), "why": f"grade: {intent}"})
    text = rubric.get("caption")
    if text:
        ops.append({"op": "overlay_text", "text": text, "position": rubric.get("caption_position", "bottom"),
                    "size": rubric.get("caption_size", 48), "color": rubric.get("caption_color", "#ffffff"),
                    "why": "caption"})
    rw = rubric.get("width")
    if rw:
        ops.append({"op": "resize", "width": int(rw), "why": "output width"})
    sharp = rubric.get("sharpen", 0.0)
    if sharp:
        ops.append({"op": "sharpen", "amount": float(sharp), "why": "crisp detail"})
    if not ops:
        # No rubric ops → emit an explicit identity grade so the schema is never
        # empty (a no-change grade is a valid, renderable operation).
        ops.append({"op": "grade",
                    "exposure": 1.0, "contrast": 0.0, "saturation": 1.0, "temperature": 0.0,
                    "why": "no change"})
    return ops


def _scale_crop(rect: dict, factor: float, inspect: dict) -> dict:
    """Same crop centre, area x factor, clamped to the frame."""
    w_img = int(inspect.get("width") or 0)
    h_img = int(inspect.get("height") or 0)
    try:
        x, y = int(rect["x"]), int(rect["y"])
        w, h = int(rect["w"]), int(rect["h"])
    except (KeyError, TypeError, ValueError):
        return dict(rect)
    if w <= 0 or h <= 0 or factor == 1.0:
        return dict(rect)
    sc = factor ** 0.5
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = max(1.0, w * sc), max(1.0, h * sc)
    nx, ny = cx - nw / 2.0, cy - nh / 2.0
    if w_img and h_img:
        nw, nh = min(nw, float(w_img)), min(nh, float(h_img))
        nx = min(max(0.0, nx), w_img - nw)
        ny = min(max(0.0, ny), h_img - nh)
    return {"x": int(round(nx)), "y": int(round(ny)), "w": int(round(nw)), "h": int(round(nh))}


def _apply_knobs(ops: list[dict], grade_f: float, crop_f: float, sharp_f: float,
                 cap_f: float, inspect: dict, rubric: dict,
                 exposure_nudge: float = 0.0, contrast_nudge: float = 0.0) -> list[dict]:
    """Post-process the legacy operations with the candidate knobs.

    With every factor at 1.0 (and no nudges) this is the identity, which is what
    keeps candidate 0 byte-identical to the pre-taste-model proposal.
    """
    out: list[dict] = []
    has_grade = any(o.get("op") == "grade" for o in ops)
    if (exposure_nudge or contrast_nudge) and not has_grade:
        # The rubric asked for no grade at all, so there is no op to nudge: add
        # one (right after a crop, else first) rather than silently ignoring the
        # candidate's only lever on the reward.
        grade_op = {"op": "grade", **_full_grade({}), "why": "explored exposure/contrast"}
        idx = 1 if ops and ops[0].get("op") == "crop" else 0
        ops = list(ops[:idx]) + [grade_op] + list(ops[idx:])
    for op in ops:
        op = dict(op)
        kind = op.get("op")
        if kind == "grade":
            silent_exposure = "exposure" not in rubric
            silent_contrast = "contrast" not in rubric
            if silent_exposure and exposure_nudge:
                op["exposure"] = round(max(0.05, float(op.get("exposure", 1.0)) + exposure_nudge), 4)
            if silent_contrast and contrast_nudge:
                op["contrast"] = round(float(op.get("contrast", 0.0)) + contrast_nudge, 4)
        if kind == "grade" and grade_f != 1.0:
            op["exposure"] = round(1.0 + grade_f * (float(op.get("exposure", 1.0)) - 1.0), 4)
            op["saturation"] = round(1.0 + grade_f * (float(op.get("saturation", 1.0)) - 1.0), 4)
            op["contrast"] = round(grade_f * float(op.get("contrast", 0.0)), 4)
            op["temperature"] = round(grade_f * float(op.get("temperature", 0.0)), 4)
        elif kind == "crop" and crop_f != 1.0:
            op.update(_scale_crop(op, crop_f, inspect))
        elif kind == "sharpen" and sharp_f != 1.0:
            op["amount"] = round(sharp_f * float(op.get("amount", 0.5)), 4)
        elif kind == "overlay_text" and cap_f != 1.0:
            op["size"] = max(1, int(round(cap_f * float(op.get("size", 48)))))
        out.append(op)
    return out


def _propose_variant(src: str, inspect: dict, rubric: dict, grade_f: float,
                     crop_f: float, sharp_f: float, cap_f: float,
                     exposure_nudge: float = 0.0, contrast_nudge: float = 0.0) -> dict:
    ops = _apply_knobs(_legacy_ops(src, inspect, rubric), grade_f, crop_f, sharp_f,
                       cap_f, inspect, rubric, exposure_nudge, contrast_nudge)
    return {"meta": {"source": src, "intent": rubric.get("intent", ""),
                     "width": inspect["width"], "height": inspect["height"],
                     "generator": "photo_core.propose"},
            "operations": ops}


def _grid_for(group: int) -> list:
    n = max(1, min(int(group), len(PHOTO_CANDIDATE_GRID)))
    return PHOTO_CANDIDATE_GRID[:n]


def _select(scores: list, rewards: list, select: str, has_identity: bool) -> tuple[int, str]:
    """argmax of the taste scores when selecting by taste, else of the rewards."""
    if select == "taste" and not has_identity:
        raise RuntimeError("select=taste requires an identity (--identity / DSH_EDITAPART_IDENTITY)")
    if has_identity and select in ("auto", "taste"):
        return max(range(len(scores)), key=lambda i: scores[i]), "taste"
    usable = [i for i, r in enumerate(rewards) if r is not None]
    if usable:
        return max(usable, key=lambda i: rewards[i]), ("objective" if len(rewards) > 1 else "only")
    return 0, "only"


def cmd_propose(src: str, inspect: dict, rubric: dict, group: int = 1,
                identity: str | None = None, style: str | None = None,
                dataset: str | None = None, clip_id: str | None = None,
                select: str = "auto", workdir: str | None = None,
                log: bool = True, score_candidates: bool | None = None) -> dict:
    """Propose one schema (group=1, legacy) or a scored GROUP of candidates.

    Candidates are scored by RENDERING them and running the objective critic
    (the photo reward is defined on the rendered image), which is why a group
    costs K magick runs. Selection then follows the same rule as video: the
    creator identity's argmax when one exists, otherwise the best objective
    reward.
    """
    tm = _taste()
    grid = _grid_for(group)
    variants: list[dict] = []
    for (grade_f, crop_f, sharp_f, cap_f, exp_nudge, con_nudge) in grid:
        schema = _propose_variant(src, inspect, rubric, grade_f, crop_f, sharp_f, cap_f,
                                  exp_nudge, con_nudge)
        knobset = {"grade_factor": grade_f, "crop_factor": crop_f,
                   "sharpen_factor": sharp_f, "caption_factor": cap_f,
                   "exposure_nudge": exp_nudge, "contrast_nudge": con_nudge}
        if any(v["schema"]["operations"] == schema["operations"] for v in variants):
            continue                      # knob had no effect on this rubric
        variants.append({"schema": schema, "knobs": knobset})
    if not variants:                      # defensive: never return an empty group
        variants = [{"schema": _propose_variant(src, inspect, rubric, 1.0, 1.0, 1.0, 1.0),
                     "knobs": {"grade_factor": 1.0, "crop_factor": 1.0, "sharpen_factor": 1.0,
                               "caption_factor": 1.0, "exposure_nudge": 0.0,
                               "contrast_nudge": 0.0}}]

    # A reward is needed to log a group or to select by objective reward. With
    # group=1 and no dataset this stays the pure legacy proposal (no render).
    if score_candidates is None:
        score_candidates = len(variants) > 1 or dataset is not None or select == "objective"
    work = workdir
    if score_candidates and not work:
        work = os.path.join(os.path.dirname(os.path.abspath(dataset)), "candidates") if dataset else None
    for i, v in enumerate(variants):
        if not score_candidates:
            v["critic"] = None
            v["render"] = None
            continue
        out_dir = work or None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, f"cand_{i}.png")
        else:
            import tempfile
            out = os.path.join(tempfile.mkdtemp(prefix="dshep_"), f"cand_{i}.png")
        cmd_render(src, v["schema"], out)
        v["critic"] = cmd_critic(v["schema"], rubric, out, None)
        v["render"] = out

    clip = clip_id or str(inspect.get("source") or src)
    scorer = tm.TasteScorer.load(identity, style) if identity else None
    spec = scorer.feature_spec if scorer else tm.FEATURE_SPEC_PHOTO
    feats = [tm.features_for(spec, v["schema"], inspect, rubric) for v in variants]
    scores = scorer.score_many(feats) if scorer else [None] * len(variants)
    rewards = [(v["critic"]["reward"] if v["critic"] else None) for v in variants]
    selected, select_by = _select(scores, rewards, select, scorer is not None)

    gid = hashlib.sha1(f"{clip}|{spec}|{[v['knobs'] for v in variants]}|{time.time_ns()}".encode()).hexdigest()[:12]
    if dataset and log:
        usable = [r for r in rewards if r is not None]
        ties = (len(set(usable)) == 1) if len(usable) > 1 else None
        _log_jsonl(dataset, {
            "kind": "group", "group_id": gid, "clip_id": clip,
            "feature_spec": spec, "created": int(time.time()), "select_by": select_by,
            "tie": bool(ties) if ties is not None else None,
            "reward_spread": (round(max(usable) - min(usable), 4) if len(usable) > 1 else None),
            "candidates": [{
                "idx": v_i, "knobs": v["knobs"], "features": feats[v_i],
                "reward_obj": (v["critic"]["reward"] if v["critic"] else None),
                "overall_obj": (v["critic"]["overall_score"] if v["critic"] else None),
                "dense_obj": (round(sum(e["reward_delta"] for e in v["critic"]["elements"]), 4)
                              if v["critic"] else None),
                "taste_score": scores[v_i],
                "n_operations": len(v["schema"]["operations"]),
            } for v_i, v in enumerate(variants)],
        })

    chosen = variants[selected]
    out_payload = {"meta": dict(chosen["schema"]["meta"]),
                   "operations": chosen["schema"]["operations"]}
    out_payload["group"] = {
        "group_id": gid, "feature_spec": spec, "size": len(variants),
        "selected": selected, "select_by": select_by, "dataset": dataset, "clip_id": clip,
        "taste": ({"identity": identity, "scores": [round(s, 6) for s in scores]}
                  if scorer else None),
        "candidates": [{
            "idx": i, "knobs": v["knobs"], "render": v["render"],
            "operations": v["schema"]["operations"],
            "reward_obj": (v["critic"]["reward"] if v["critic"] else None),
            "overall_obj": (v["critic"]["overall_score"] if v["critic"] else None),
            "taste_score": (round(scores[i], 6) if scores[i] is not None else None),
        } for i, v in enumerate(variants)],
    }
    return out_payload


def _fraction_crop(crop: dict, inspect: dict) -> dict:
    w, h = inspect["width"], inspect["height"]
    if "percent" in crop:
        p = crop["percent"]
        return {"x": int(w * p[0]), "y": int(h * p[1]), "w": int(w * p[2]), "h": int(h * p[3])}
    return {"x": int(crop.get("x", 0)), "y": int(crop.get("y", 0)),
            "w": int(crop.get("w", w)), "h": int(crop.get("h", h))}


def _full_grade(g: dict) -> dict:
    return {"exposure": g.get("exposure", 1.0), "contrast": g.get("contrast", 0.0),
            "saturation": g.get("saturation", 1.0), "temperature": g.get("temperature", 0.0)}


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def cmd_render(src: str, schema: dict, out: str) -> dict:
    _ensure_magick()
    ops = schema.get("operations", [])
    cmd = [MAGICK, src]
    for op in ops:
        kind = op.get("op")
        if kind == "crop":
            cmd += ["-crop", f"{op['w']}x{op['h']}+{op['x']}+{op['y']}", "+repage"]
        elif kind == "grade":
            g = op
            cmd += _grade_args(g)
        elif kind == "overlay_text":
            cmd += _annotate_args(op)
        elif kind == "resize":
            cmd += ["-resize", f"{op['width']}x"]
        elif kind == "sharpen":
            cmd += ["-sharpen", f"0x{op.get('amount', 0.5)}"]
        else:
            continue
    cmd += [out]
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"magick render failed: {r.stderr[:400]}")
    d = _dims(out)
    return {"out": out, "width": d[0], "height": d[1]}


def _grade_args(g: dict) -> list[str]:
    args: list[str] = []
    sat = float(g.get("saturation", 1.0))
    exp = float(g.get("exposure", 1.0))
    # IM modulate: brightness%, saturation%, hue% (exposure -> brightness).
    args += ["-modulate", f"{int(round(exp * 100))},{int(round(sat * 100))},100"]
    con = float(g.get("contrast", 0.0))
    if con:
        args += ["-sigmoidal-contrast", f"{con}x50%"]
    temp = float(g.get("temperature", 0.0))
    if temp:
        # temperature>0 warms (R up, B down); <0 cools.
        r = int(round(255 * temp))
        args += ["-channel", "RGB",
                 "-fill", f"rgb({int(255 + r)},255,{int(255 - r)})",
                 "-colorize", "0",
                 "-channel", "RGB"]
    return args


def _escape_annotate_text(text: str) -> str:
    """Neutralize ImageMagick text-interpretation metacharacters in a caption.

    `-annotate`/`-caption`/`-label` parse `%[...]`/`%[fx:...]` format escapes and
    a leading `@` as "read text from file". Unescaped untrusted caption text
    could therefore run an IM expression or trigger a file read (a real issue on
    hosts whose ImageMagick policy does NOT disable `@`/`%`). Rather than depend
    on IM's version-specific escape grammar, replace the dangerous metacharacters
    with visually-identical non-ASCII codepoints that IM renders literally and
    never interprets:  '%' -> '％' (fullwidth) and '@' -> '＠' (fullwidth). This is
    defense-in-depth and works regardless of the ImageMagick policy.xml.
    """
    if not text:
        return ""
    # Fullwidth percent and at-sign render identically to the ASCII forms but are
    # distinct codepoints, so IM's format parser and @-file-read never trigger.
    return (
        text.replace("%", "\uff05")  # FULLWIDTH PERCENT SIGN
            .replace("@", "\uff20")  # FULLWIDTH COMMERCIAL AT
            .replace("\\", "\uff3c")  # FULLWIDTH REVERSE SOLIDUS (defensive)
    )


def _annotate_args(op: dict) -> list[str]:
    size = op.get("size", 48)
    color = op.get("color", "#ffffff")
    text = _escape_annotate_text(op.get("text", ""))
    return ["-gravity", "South", "-fill", color, "-pointsize", str(size),
            "-annotate", "+0+40", text]


# --------------------------------------------------------------------------
# critic
# --------------------------------------------------------------------------
def _score_objective(schema: dict, rubric: dict, result_path: str,
                     subjective: float | None) -> dict:
    insp = cmd_inspect(result_path)
    target_luma = float(rubric.get("target_luma", 0.5) or 0.5)
    tol = float(rubric.get("exposure_tol", 0.15) or 0.15)
    gap = abs(insp["mean_luma"] - target_luma)
    exposure_ok = gap <= tol
    # GRADED, not thresholded. `exposure_ok` is still reported, but the score has
    # a linear falloff beyond the tolerance: a pure threshold makes every
    # candidate that misses it score identically, so a candidate GROUP ties and
    # the group-relative advantage is exactly zero (measured: 6/6 candidates at
    # reward -0.05, spread 0.0, before this change). Full credit inside the
    # tolerance is unchanged, so passing candidates score as they always did.
    exposure_score = 1.0 if exposure_ok else max(0.0, 1.0 - (gap - tol) / max(1e-9, tol))
    min_contrast = float(rubric.get("min_contrast", 0.12) or 0.12)
    contrast_ok = insp["luma_std"] >= min_contrast
    contrast_score = (1.0 if contrast_ok
                      else min(1.0, max(0.0, insp["luma_std"] / max(1e-9, min_contrast))))
    elements = []
    for op in schema.get("operations", []):
        deltas = []
        kind = op.get("op")
        if kind == "grade" and not exposure_ok:
            deltas.append(f"exposure {insp['mean_luma']:.2f} off target {target_luma:.2f}")
        if kind == "resize" and rubric.get("width") and op.get("width") != rubric["width"]:
            deltas.append(f"resize {op.get('width')} != target {rubric['width']}")
        elements.append({
            "op": kind, "decision": "drop" if deltas else "keep",
            "why": "; ".join(deltas) if deltas else (op.get("why", "") or "passes rubric"),
            "reward_delta": round(-0.2 * len(deltas), 2),
        })
    dense = sum(e["reward_delta"] for e in elements)
    metrics = {"mean_luma": insp["mean_luma"], "luma_std": insp["luma_std"],
               "saturation": insp["saturation"], "exposure_ok": exposure_ok,
               "contrast_ok": contrast_ok, "exposure_score": round(exposure_score, 3),
               "contrast_score": round(contrast_score, 3)}
    overall = 0.5 * (exposure_score + contrast_score) / 2.0 + 0.5 * dense
    if subjective is not None:
        overall = 0.6 * overall + 0.4 * subjective
    return {"overall_score": round(overall, 3), "metrics": metrics,
            "reward": round(overall + dense, 3), "elements": elements,
            "revise": [e["op"] + "#" + str(i) for i, e in enumerate(elements) if e["decision"] == "drop"][:8]}


def cmd_critic(schema: dict, rubric: dict, result_path: str, subjective: float | None,
               dataset: str | None = None, group_id: str | None = None,
               candidate: int | None = None, chosen_by: str = "critic") -> dict:
    """Score a rendered photo, and optionally log the outcome for taste training.

    `chosen_by=creator|agent` marks a REVEALED preference: the pick is what the
    creator/agent actually wanted, so it becomes that group's top reward and a
    preference target. Without a user-dependent term in the objective, per-user
    taste is unidentifiable (see docs/paper-findings.md, Finding 1).
    """
    out = _score_objective(schema, rubric, result_path, subjective)
    if dataset and group_id:
        idx = int(candidate or 0)
        _log_jsonl(dataset, {
            "kind": "reward", "group_id": group_id, "candidate": idx,
            "chosen_by": chosen_by, "reward": out["reward"],
            "objective_reward": out["reward"], "overall_obj": out["overall_score"],
            "dense_obj": round(sum(e["reward_delta"] for e in out["elements"]), 4),
            "subjective": subjective, "render": result_path, "logged_at": int(time.time()),
        })
        out["logged"] = {"dataset": dataset, "group_id": group_id, "candidate": idx,
                         "chosen_by": chosen_by, "reward": out["reward"]}
    return out


# --------------------------------------------------------------------------
# revise
# --------------------------------------------------------------------------
def cmd_revise(schema: dict, critic: dict) -> dict:
    drop_idx = set()
    for rev in critic.get("revise", []):
        try:
            drop_idx.add(int(rev.rsplit("#", 1)[1]))
        except (IndexError, ValueError):
            pass
    kept = [op for i, op in enumerate(schema.get("operations", [])) if i not in drop_idx]
    if not kept:
        kept = schema.get("operations", [])
    schema["operations"] = kept
    schema["meta"]["revised_by"] = "photo-critic"
    return schema


# --------------------------------------------------------------------------
# taste model (same code path as the video core, photo/v1 defaults)
# --------------------------------------------------------------------------
def cmd_taste_train(dataset: str, identity: str, style: str | None = None,
                    creator: str = "default", epochs: int = 150, lr_muon: float = 0.005,
                    lr_adamw: float = 0.005, temperature: float = 0.5, clip_eps: float = 0.2,
                    entropy_coef: float = 0.01, pref_coef: float = 0.5,
                    lambda_dense: float = 1.0, grad_clip: float = 1.0, inner_steps: int = 1,
                    batch_groups: int = 16, revealed_pref_bonus: float = 1.0,
                    seed: int = 0, holdout_every: int = 5, warm_start: bool = True,
                    freeze_style: bool = False, verbose: bool = False) -> dict:
    return _taste().train(
        dataset, identity, style=style, creator=creator, epochs=int(epochs),
        lr_muon=lr_muon, lr_adamw=lr_adamw, temperature=temperature, clip_eps=clip_eps,
        entropy_coef=entropy_coef, pref_coef=pref_coef, lambda_dense=lambda_dense,
        grad_clip=grad_clip, inner_steps=int(inner_steps), batch_groups=int(batch_groups),
        revealed_pref_bonus=revealed_pref_bonus, seed=int(seed),
        holdout_every=int(holdout_every), warm_start=warm_start,
        freeze_style=freeze_style, verbose=verbose)


def cmd_identity_init(identity: str, style: str | None = None, creator: str = "default",
                      spec: str | None = None) -> dict:
    tm = _taste()
    return tm.init_identity(identity, style=style, creator=creator,
                            spec=spec or tm.FEATURE_SPEC_PHOTO)


def cmd_taste_status(identity: str | None = None, dataset: str | None = None,
                     style: str | None = None) -> dict:
    tm = _taste()
    out: dict = {"identity": None, "dataset": None}
    if identity:
        if os.path.exists(identity):
            out["identity"] = {**tm.TasteScorer.load(identity, style).info(), "exists": True}
        else:
            out["identity"] = {"path": identity, "exists": False,
                               "hint": "run identity_init (or taste_train) to create it"}
    if dataset:
        out["dataset"] = tm.dataset_status(dataset)
    return out


def cmd_taste(identity: str, style: str | None = None, features: str | None = None,
              schema: str | None = None, inspect: str | None = None,
              rubric: str | None = None) -> dict:
    tm = _taste()
    scorer = tm.TasteScorer.load(identity, style)
    if features:
        payload = _fixture(features)
        if isinstance(payload, list):
            return {"scores": scorer.score_many(payload), "feature_spec": scorer.feature_spec}
        return {"score": scorer.score_features(payload), "feature_spec": scorer.feature_spec}
    if not (schema and inspect and rubric):
        raise RuntimeError("taste needs --features or all of --schema/--inspect/--rubric")
    feats = tm.features_for(scorer.feature_spec, _fixture(schema), _fixture(inspect),
                            _fixture(rubric))
    return {"score": scorer.score_features(feats), "features": feats,
            "feature_spec": scorer.feature_spec}


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect"); p.add_argument("src")
    p = sub.add_parser("propose"); p.add_argument("src"); p.add_argument("inspect"); p.add_argument("rubric")
    p.add_argument("--group", type=int, default=int(os.getenv("DSH_EDITAPART_GROUP", "1")),
                   help="candidates to consider (1 = legacy single schema)")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"))
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))
    p.add_argument("--dataset", default=os.getenv("DSH_EDITAPART_DATASET"))
    p.add_argument("--clip-id", default=None)
    p.add_argument("--select", choices=["auto", "taste", "objective"], default="auto")
    p.add_argument("--workdir", default=None,
                   help="where candidate renders are written (default: <dataset dir>/candidates)")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-score-candidates", action="store_true",
                   help="do not render/score candidates (selection cannot use rewards)")
    p = sub.add_parser("render"); p.add_argument("src"); p.add_argument("schema"); p.add_argument("out")
    p = sub.add_parser("critic"); p.add_argument("schema"); p.add_argument("rubric"); p.add_argument("result")
    p.add_argument("--subjective", type=float, default=None)
    p.add_argument("--dataset", default=os.getenv("DSH_EDITAPART_DATASET"))
    p.add_argument("--group-id", default=None)
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--chosen-by", choices=["critic", "creator", "agent", "human"],
                   default="critic")
    p = sub.add_parser("revise"); p.add_argument("schema"); p.add_argument("critic")

    p = sub.add_parser("taste_train")
    p.add_argument("--dataset", default=os.getenv("DSH_EDITAPART_DATASET"), required=False)
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"), required=False)
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))
    p.add_argument("--creator", default=os.getenv("DSH_EDITAPART_CREATOR", "default"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr-muon", type=float, default=0.005)
    p.add_argument("--lr-adamw", type=float, default=0.005)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--pref-coef", type=float, default=0.5)
    p.add_argument("--lambda-dense", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--batch-groups", type=int, default=16)
    p.add_argument("--revealed-pref-bonus", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-every", type=int, default=5)
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--freeze-style", action="store_true",
                   help="train only z_u against a frozen shared style-brain")
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("identity_init")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"), required=False)
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))
    p.add_argument("--creator", default=os.getenv("DSH_EDITAPART_CREATOR", "default"))
    p.add_argument("--spec", default=None, help="default photo/v1")

    p = sub.add_parser("taste_status")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"))
    p.add_argument("--dataset", default=os.getenv("DSH_EDITAPART_DATASET"))
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))

    p = sub.add_parser("taste")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"), required=False)
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))
    p.add_argument("--features", default=None)
    p.add_argument("--schema", default=None)
    p.add_argument("--inspect", default=None)
    p.add_argument("--rubric", default=None)

    args = ap.parse_args()
    try:
        if args.cmd == "inspect":
            result = cmd_inspect(args.src)
        elif args.cmd == "propose":
            result = cmd_propose(args.src, json.loads(args.inspect), json.loads(args.rubric),
                                 group=args.group, identity=args.identity, style=args.style,
                                 dataset=args.dataset, clip_id=args.clip_id, select=args.select,
                                 workdir=args.workdir, log=not args.no_log,
                                 score_candidates=(False if args.no_score_candidates else None))
        elif args.cmd == "render":
            result = cmd_render(args.src, json.loads(args.schema), args.out)
        elif args.cmd == "critic":
            result = cmd_critic(json.loads(args.schema), json.loads(args.rubric), args.result,
                                args.subjective, dataset=args.dataset, group_id=args.group_id,
                                candidate=args.candidate, chosen_by=args.chosen_by)
        elif args.cmd == "revise":
            result = cmd_revise(json.loads(args.schema), json.loads(args.critic))
        elif args.cmd == "taste_train":
            if not args.dataset or not args.identity:
                raise RuntimeError("taste_train needs --dataset and --identity "
                                   "(or DSH_EDITAPART_DATASET / DSH_EDITAPART_IDENTITY)")
            result = cmd_taste_train(args.dataset, args.identity, style=args.style,
                                     creator=args.creator, epochs=args.epochs,
                                     lr_muon=args.lr_muon, lr_adamw=args.lr_adamw,
                                     temperature=args.temperature, clip_eps=args.clip_eps,
                                     entropy_coef=args.entropy_coef, pref_coef=args.pref_coef,
                                     lambda_dense=args.lambda_dense, grad_clip=args.grad_clip,
                                     inner_steps=args.inner_steps,
                                     batch_groups=args.batch_groups,
                                     revealed_pref_bonus=args.revealed_pref_bonus,
                                     seed=args.seed, holdout_every=args.holdout_every,
                                     warm_start=not args.no_warm_start,
                                     freeze_style=args.freeze_style, verbose=args.verbose)
        elif args.cmd == "identity_init":
            if not args.identity:
                raise RuntimeError("identity_init needs --identity (or DSH_EDITAPART_IDENTITY)")
            result = cmd_identity_init(args.identity, style=args.style, creator=args.creator,
                                       spec=args.spec)
        elif args.cmd == "taste_status":
            result = cmd_taste_status(args.identity, args.dataset, args.style)
        elif args.cmd == "taste":
            if not args.identity:
                raise RuntimeError("taste needs --identity (or DSH_EDITAPART_IDENTITY)")
            result = cmd_taste(args.identity, style=args.style, features=args.features,
                               schema=args.schema, inspect=args.inspect, rubric=args.rubric)
        else:
            ap.error(f"unknown command {args.cmd}")
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
