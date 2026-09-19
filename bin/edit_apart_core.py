#!/usr/bin/env python3
"""Shared engine core for the AI Taste Video Editor preset plugin.

This module is the single implementation of the render->critique->revise loop
that the preset's plugin (`plugins/video-editor.mjs`) shells out to from its
tool `execute` handlers. It is deliberately import-free (stdlib + subprocess
only) so the plugin never needs to import the harness, and deterministic.

The plugin invokes it as:
    edit_apart_core.py <subcommand> <args...>

Subcommands mirror the tools: inventory | features | propose | render |
review_frames | critic | revise | train | taste | taste_features | taste_status |
identity_init | identity_info. Each reads JSON on stdin and writes JSON to
stdout, so the plugin can pass args through and surface the result directly.

The taste model (bin/taste_model.py) is wired in HERE, in the loop itself:

  * `propose` builds a GROUP of candidate schemas from a deterministic grid,
    scores each with the creator identity when one is configured, selects a
    candidate, and appends the group to the creator's training log.
  * `critic` can append the render + vision outcome for the selected candidate
    (that is the reward the trainer consumes).
  * `train` fits the shared style-brain + the creator latent (Muon on 2D, AdamW
    on z_u and 1D) and writes the per-creator GGUF.
  * `render` / `review_frames` / `revise` are untouched.

Every piece is optional and off unless asked for: with no identity and
`group=1` the returned `structure`/`meta`/`globals` are exactly what the engine
returned before the taste model was wired in (an additive `group` key is the
only difference).
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

# Toolchains are resolved at runtime, NOT hard-locked. A deployment may set
# DSH_FFMPEG / DSH_FFPROBE / DSH_SCENEDETECT / DSH_EDIT_PY, or rely on PATH, or
# (web profile) supply an in-browser ffmpeg / scenedetect through a store plugin.
# Each resolver is env-first (DSH_*), then PATH (shutil.which), then a
# predictable name fallback — so the core is NOT a hard dependency on one
# absolute binary path. The absolute venv path is only the last resort, and
# only when env + PATH both miss.
def _resolve(env_key: str, name: str, fallback: str | None = None) -> str:
    env = os.getenv(env_key)
    if env:
        return env
    found = shutil.which(name)
    if found:
        return found
    return fallback or name


# NOTE: the engine never spawns python itself — it runs scenedetect/ffmpeg/
# ffprobe as subprocesses. The python interpreter that LAUNCHES edit_apart_core.py is
# the plugin's concern (see plugins/video-editor.mjs), not this module's.
VENV_SCENEDETECT = _resolve("DSH_SCENEDETECT", "scenedetect")
FFMPEG = _resolve("DSH_FFMPEG", "ffmpeg")
FFPROBE = _resolve("DSH_FFPROBE", "ffprobe")


# --------------------------------------------------------------------------
# taste-model bridge
# --------------------------------------------------------------------------
def _taste():
    """Import bin/taste_model.py (sitting next to this file), lazily.

    Lazy on purpose: `inventory`/`features`/`render`/`review_frames`/`revise` and
    `propose group=1` must keep working on a machine with no numpy at all.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import taste_model  # noqa: PLC0415 — deliberate lazy import
    return taste_model


def _log_jsonl(path: str, record: dict) -> None:
    """Append one record to an append-only JSONL log (fsynced: the loop's
    training data must survive a crash mid-session)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _fixture(value: str) -> dict:
    """A JSON object given either inline or as a path."""
    if isinstance(value, dict):
        return value
    if os.path.exists(value):
        with open(value, encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(value)


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------
def cmd_inventory(src: str, threshold: float, min_scene_len: float) -> dict:
    """scenedetect cut detection -> shot list (0.7.1 syntax)."""
    out_dir = temp_dir()
    out_file = os.path.join(out_dir, "_sd_scenes.csv")
    r = subprocess.run(
        [VENV_SCENEDETECT, "-i", src, "detect-content",
         "--threshold", str(threshold), "--min-scene-len", str(min_scene_len),
         "list-scenes", "--output", out_dir, "--filename", "_sd_scenes.csv"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"scenedetect failed: {r.stdout} {r.stderr}")
    if not os.path.exists(out_file):
        raise RuntimeError(f"no scene CSV produced at {out_file}")
    with open(out_file, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    header_i = next(i for i, ln in enumerate(lines) if ln.lower().startswith("scene number"))
    header = [h.strip() for h in lines[header_i].split(",")]

    def col(name_base: str) -> int:
        # Exact match on the column name (case-insensitive). Substring matching
        # is fragile: "Start Time" would also match "Start Timecode", and the
        # wrong column would be picked. Require an exact match on the full name.
        for i, h in enumerate(header):
            if h.strip().lower() == name_base.strip().lower():
                return i
        raise KeyError(f"{name_base!r} not in header {header}")

    start_i = col("Start Time (seconds)")
    end_i = col("End Time (seconds)")
    shots = []
    for idx, ln in enumerate(lines[header_i + 1:], start=1):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) <= max(start_i, end_i):
            continue
        start = float(parts[start_i])
        end = float(parts[end_i])
        shots.append({"shot": f"clip_{idx:03d}", "start": round(start, 3),
                      "end": round(end, 3), "duration": round(end - start, 3),
                      "src_idx": idx})
    return {"source": src, "shots": shots}


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    k = min(len(sv) - 1, int(round(p * (len(sv) - 1))))
    return sv[k]


def _luma(src: str, start: float, end: float) -> list[float]:
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{start:g}", "-t", f"{max(0.05, end - start):g}",
         "-i", src, "-vf", "fps=4,scale=8:8,format=gray", "-f", "rawvideo",
         "-pix_fmt", "gray", "-"], capture_output=True)
    vals: list[float] = []
    if r.returncode == 0 and r.stdout:
        n = 64
        for i in range(0, len(r.stdout), n):
            chunk = r.stdout[i:i + n]
            if chunk:
                vals.append(sum(chunk) / len(chunk))
    return vals


def _motion(luma: list[float]) -> float:
    if len(luma) < 2:
        return 0.0
    return sum(abs(luma[i] - luma[i - 1]) for i in range(1, len(luma))) / (len(luma) - 1)


def _audio(src: str, start: float, end: float) -> dict:
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{start:g}", "-t", f"{max(0.05, end - start):g}",
         "-i", src, "-af", "astats=reset=1:metadata=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:key=lavfi.astats.Overall.Peak_level",
         "-f", "null", "-"], capture_output=True, text=True)
    rms = peak = 0.0
    for line in r.stdout.splitlines():
        if "RMS_level" in line and "=" in line:
            try:
                rms = float(line.split("=", 1)[1].strip())
            except ValueError:
                rms = 0.0
        if "Peak_level" in line and "=" in line:
            try:
                peak = float(line.split("=", 1)[1].strip())
            except ValueError:
                peak = 0.0
    return {"rms_db": round(rms, 2), "peak_db": round(peak, 2)}


def cmd_features(src: str, inventory: dict) -> dict:
    for shot in inventory["shots"]:
        start, end = shot["start"], shot["end"]
        luma = _luma(src, start, end)
        feat = {"lum_p50": round(_percentile(luma, 0.5), 1),
                "lum_p95": round(_percentile(luma, 0.95), 1),
                "motion": round(_motion(luma), 2)}
        try:
            feat.update(_audio(src, start, end))
        except Exception:
            feat.update({"rms_db": 0.0, "peak_db": 0.0})
        shot["features"] = feat
    return inventory


# --------------------------------------------------------------------------
# propose (now a GROUP proposal: the candidates GRPO learns to rank)
# --------------------------------------------------------------------------
# Deterministic candidate grid. Candidate 0 is the rubric verbatim, so group=1
# reproduces the legacy single-schema proposal. The other candidates trade off
# target duration and the minimum-shot filter — exactly the v0 scope (pacing +
# shot selection + tempo) the taste model is allowed to learn. No randomness:
# the same inventory + rubric always yields the same group.
CANDIDATE_GRID = [
    (1.0, 1.0, True, True),     # legacy: rubric verbatim
    (0.75, 1.0, True, True),    # shorter cut
    (1.25, 1.0, True, True),    # longer cut
    (1.0, 0.6, True, True),     # let shorter shots in
    (1.0, 1.4, True, True),     # stricter minimum shot length
    (0.6, 1.0, True, True),     # very short cut
    (1.5, 1.0, True, True),     # very long cut
    (0.8, 0.7, False, True),    # no short-shot filter: keep everything
]


def _propose_variant(inventory: dict, rubric: dict, target_factor: float,
                     min_factor: float, skip_short: bool, keep_max: bool) -> dict:
    """The legacy proposal algorithm, parameterized by the candidate knobs.

    With (1.0, 1.0, True, True) this returns exactly what the old `cmd_propose`
    returned for the same inventory + rubric.
    """
    shots = inventory["shots"]
    base_min = rubric.get("min_shot_dur", 0.8)
    base_max = rubric.get("max_shot_dur")
    base_target = rubric.get("target_duration")
    min_dur = base_min * min_factor
    max_dur = base_max if keep_max else None
    target = base_target * target_factor if base_target else None
    kept = [s for s in shots
            if (not skip_short or s["duration"] >= min_dur)
            and (max_dur is None or s["duration"] <= max_dur)]
    if not kept:
        raise RuntimeError("rubric filtered every shot; relax min/max shot duration")
    structure = []
    for s in kept:
        structure.append({
            "shot": s["shot"], "trim": {"in": s["start"], "out": s["end"]},
            "retime": 0.0, "timeline": {"in": 0.0, "out": round(s["end"] - s["start"], 3)},
            "transition": {"type": "cut", "dur": 0.0, "params": {}},
            "transform": {"scale": 1.0, "crop": None, "position": None},
            "grade": {"lut": None, "eq": {}}, "audio": {"level": 1.0, "duck": 0.0, "fade": 0.0},
            "overlay": [], "why": "selected; respects rubric pacing",
        })
    if target and sum(s["trim"]["out"] - s["trim"]["in"] for s in structure) > target:
        run = 0.0
        for i, s in enumerate(structure):
            d = s["trim"]["out"] - s["trim"]["in"]
            if run + d >= target:
                s["timeline"]["out"] = round(max(0, target - run), 3)
                s["trim"]["out"] = round(s["trim"]["in"] + (target - run), 3)
                structure = structure[:i + 1]
                break
            run += d
    return {"meta": {"source": inventory.get("source", ""),
                     "intent": rubric.get("intent", ""),
                     "generator": "edit_apart_core.propose"},
            "structure": structure,
            "globals": {"color": None, "audioMix": {}, "music": {"bed": None, "syncToBeat": False},
                        "pace": rubric.get("pace")}}


def _grid_for(group: int) -> list:
    """The deterministic candidate grid, capped at the table size."""
    n = max(1, min(int(group), len(CANDIDATE_GRID)))
    return CANDIDATE_GRID[:n]


def cmd_propose(inventory: dict, rubric: dict, group: int = 1, identity: str | None = None,
                style: str | None = None, dataset: str | None = None,
                clip_id: str | None = None, select: str = "auto",
                log: bool = True) -> dict:
    """Propose a group of candidate schemas, select one, and log the group.

    Selection: with an identity, `select=auto|taste` picks the taste-model
    argmax; otherwise (or with `select=objective`) it picks the best objective
    reward. Either way `group=1` selects the only candidate, which is the legacy
    schema.
    """
    tm = _taste()
    grid = _grid_for(group)
    clip = clip_id or str(inventory.get("source") or "")
    cands = []
    for i, (target_factor, min_factor, skip_short, keep_max) in enumerate(grid):
        schema = _propose_variant(inventory, rubric, target_factor, min_factor,
                                  skip_short, keep_max)
        crit = _score_objective(schema, inventory, rubric, None)
        cands.append({
            "idx": i, "schema": schema, "critic": crit,
            "knobs": {"target_factor": target_factor, "min_shot_factor": min_factor,
                      "skip_short": skip_short, "keep_max": keep_max},
        })

    scorer = None
    spec = None
    if identity:
        scorer = tm.TasteScorer.load(identity, style)
        spec = scorer.feature_spec
    if spec is None:
        spec = tm.FEATURE_SPEC_VIDEO
    feats = [tm.features_for(spec, c["schema"], inventory, rubric) for c in cands]
    scores = scorer.score_many(feats) if scorer else [None] * len(cands)

    if select == "taste" and scorer is None:
        raise RuntimeError("select=taste requires an identity (--identity / DSH_EDITAPART_IDENTITY)")
    if scorer is not None and select in ("auto", "taste"):
        selected = max(range(len(cands)), key=lambda i: scores[i])
        select_by = "taste"
    else:
        selected = max(range(len(cands)), key=lambda i: cands[i]["critic"]["reward"])
        select_by = "objective" if len(cands) > 1 else "only"

    gid = hashlib.sha1(
        f"{clip}|{spec}|{[c['knobs'] for c in cands]}|{time.time_ns()}".encode()
    ).hexdigest()[:12]
    if dataset and log:
        _log_jsonl(dataset, {
            "kind": "group", "group_id": gid, "clip_id": clip,
            "feature_spec": spec, "created": int(time.time()),
            "select_by": select_by,
            "candidates": [{
                "idx": c["idx"], "knobs": c["knobs"], "features": feats[c["idx"]],
                "reward_obj": c["critic"]["reward"], "taste_score": scores[c["idx"]],
                "overall_obj": c["critic"]["overall_score"],
                "dense_obj": round(sum(e["reward_delta"] for e in c["critic"]["elements"]), 4),
                "n_segments": len(c["schema"]["structure"]),
            } for c in cands],
        })

    chosen = cands[selected]
    out = {"meta": dict(chosen["schema"]["meta"]), "structure": chosen["schema"]["structure"],
           "globals": chosen["schema"]["globals"]}
    out["group"] = {
        "group_id": gid, "feature_spec": spec, "size": len(cands),
        "selected": selected, "select_by": select_by, "dataset": dataset,
        "clip_id": clip,
        "taste": ({"identity": identity, "scores": [round(s, 6) for s in scores]}
                  if scorer else None),
        "candidates": [{
            "idx": c["idx"], "knobs": c["knobs"],
            "reward_obj": c["critic"]["reward"],
            "overall_obj": c["critic"]["overall_score"],
            "taste_score": (round(scores[c["idx"]], 6) if scores[c["idx"]] is not None else None),
            "n_segments": len(c["schema"]["structure"]),
            "duration": round(sum(s["trim"]["out"] - s["trim"]["in"]
                                  for s in c["schema"]["structure"]), 3),
        } for c in cands],
    }
    return out


# --------------------------------------------------------------------------
# render (mirrors render_from_schema.py graph builder)
# --------------------------------------------------------------------------
def _probe(src: str) -> dict:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,duration",
                        "-of", "json", src], capture_output=True, text=True)
    data = json.loads(r.stdout)
    return next((s for s in data["streams"] if s["codec_type"] == "video"), {})


def cmd_render(src: str, schema: dict, out: str) -> dict:
    segs = schema.get("structure", [])
    if not segs:
        raise RuntimeError("schema has no structure segments")
    # A segment may come from the SOURCE footage (default) or from an
    # AI-GENERATED clip (the store/harness generation tool produced an asset).
    # Build the input list: input 0 = source, then one per distinct generated
    # asset. This is how a generated clip is INSERTED into a regular edit.
    inputs: list[str] = [src]
    asset_index: dict[str, int] = {}

    def src_index(seg: dict) -> int:
        if not seg.get("generated"):
            return 0
        asset = seg.get("asset")
        if not asset:
            raise RuntimeError("generated segment is missing its asset path")
        if asset not in asset_index:
            asset_index[asset] = len(inputs)
            inputs.append(asset)
        return asset_index[asset]

    v_filter: list[str] = []
    in_labels: list[str] = []
    for i, seg in enumerate(segs):
        s = src_index(seg)
        t_in, t_out = seg["trim"]["in"], seg["trim"]["out"]
        parts = ["trim=" + f"start={t_in:g}:end={t_out:g}", "setpts=PTS-STARTPTS"]
        retime = seg.get("retime", 0.0)
        if retime not in (0.0, None):
            parts.append(f"setpts=PTS/{retime:g}")
        tf = seg.get("transform") or {}
        scale = tf.get("scale", 1.0)
        if scale and scale != 1.0:
            parts.append(f"scale=iw*{scale:g}:ih*{scale:g}")
        crop = tf.get("crop")
        if crop:
            parts.append(f"crop={crop}")
        grade = seg.get("grade") or {}
        eq = grade.get("eq") or {}
        eqopts = [f"{k}={eq[k]}" for k in ("brightness", "contrast", "saturation", "gamma") if k in eq]
        if eqopts:
            parts.append("eq=" + ":".join(eqopts))
        v_filter.append(f"[{s}:v]" + ",".join(parts) + f",format=yuv420p[v{i}]")
        in_labels.append(f"[v{i}]")
    concat_v = "".join(in_labels) + f"concat=n={len(segs)}:v=1:a=0[outv]"

    a_filter: list[str] = []
    a_labels: list[str] = []
    for i, seg in enumerate(segs):
        s = src_index(seg)
        t_in, t_out = seg["trim"]["in"], seg["trim"]["out"]
        adur = t_out - t_in
        a = f"[{s}:a]" + f"atrim=start={t_in:g}:end={t_out:g},asetpts=PTS-STARTPTS"
        au = seg.get("audio") or {}
        lvl = au.get("level", 1.0)
        if lvl and lvl != 1.0:
            a += f",volume={lvl:g}"
        fade = au.get("fade", 0.0)
        if fade:
            a += f",afade=t=in:st=0:d={fade:g},afade=t=out:st={max(0, adur - fade):g}:d={fade:g}"
        a += f"[a{i}]"
        a_filter.append(a)
        a_labels.append(f"[a{i}]")
    if a_labels:
        a_chain = "".join(a_labels) + f"concat=n={len(a_labels)}:v=0:a=1,loudnorm[aout]"
    else:
        a_chain = "anullsrc=channel_layout=stereo:sample_rate=48000[aout]"
    fc = ";".join(v_filter + [concat_v]) + ";" + ";".join(a_filter + [a_chain])

    input_args: list[str] = []
    for inp in inputs:
        input_args += ["-i", inp]
    cmd = [FFMPEG, "-y", *input_args, "-filter_complex", fc,
           "-map", "[outv]", "-map", "[aout]", "-c:v", "libx264", "-preset",
           "veryfast", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed: {r.stderr[:400]}")
    return {"out": out, "duration": _probe(out).get("duration")}


# --------------------------------------------------------------------------
# review_frames
# --------------------------------------------------------------------------
def cmd_review_frames(src: str, schema: dict, outdir: str, size: int) -> dict:
    os.makedirs(outdir, exist_ok=True)
    manifest = []
    for seg in schema.get("structure", []):
        shot = seg["shot"]
        t = (seg["trim"]["in"] + seg["trim"]["out"]) / 2.0
        fp = os.path.join(outdir, f"frame_{shot}.png")
        r = subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{t:g}",
                            "-i", src, "-frames:v", "1", "-vf", f"scale={size}:-2", fp],
                           capture_output=True, text=True)
        if r.returncode == 0:
            manifest.append({"shot": shot, "frame": fp, "t": round(t, 3), "trim": seg["trim"]})
    return {"outdir": outdir, "manifest": manifest}


# --------------------------------------------------------------------------
# critic
# --------------------------------------------------------------------------
def _seg_id(s: dict, idx: int) -> str:
    """Stable id for a schema segment: a source shot name, a generated-asset key,
    or a positional fallback.

    Determinism matters because `cmd_critic` and `cmd_revise` run in SEPARATE
    processes: the id the critic emits must be recomputable by revise from the
    same schema. A `str(id(s))` fallback would be a memory address that differs
    across processes, so a drop would silently never match. Position is stable
    for a given schema, so an index-based fallback is safe.
    """
    if s.get("generated"):
        t = s.get("trim") or {}
        return f"gen:{s.get('asset', '?')}#{t.get('in', 0)}:{t.get('out', 0)}"
    if s.get("shot"):
        return str(s["shot"])
    return f"seg#{idx}"


def _score_objective(schema: dict, inventory: dict, rubric: dict,
                     subjective: float | None) -> dict:
    """The objective + dense-delta critic (arithmetic only; no render needed).

    Also the reward source for the taste model: `reward` (= overall + dense) is
    the dense-shaped reward GRPO group-normalizes.
    """
    segs = schema.get("structure", [])
    durs = [s["trim"]["out"] - s["trim"]["in"] for s in segs]
    total = sum(durs)
    cpm = len(segs) / (total / 60.0) if total else 0.0
    mean_d = sum(durs) / len(durs) if durs else 0.0
    min_d = min(durs) if durs else 0.0
    max_d = max(durs) if durs else 0.0
    target = rubric.get("target_duration")
    target_ok = (target is None) or (abs(total - target) <= 1.0)
    pace = rubric.get("pace")
    band = {"slow": (2.0, 6.0), "medium": (1.0, 3.0), "brisk": (0.8, 2.5)}.get(pace, (0.8, 4.0))
    in_band = band[0] <= mean_d <= band[1]
    score_sig = (0.4 if target_ok else 0.0) + (0.3 if in_band else 0.0) \
        + (0.3 if (not durs or min_d >= rubric.get("no_shot_under_s", 0.9)) else 0.0)
    metrics = {"cpm": round(cpm, 1), "mean_shot_dur": round(mean_d, 2),
               "min_shot_dur": round(min_d, 2), "max_shot_dur": round(max_d, 2),
               "target_ok": target_ok, "in_band": in_band, "score_sig": round(score_sig, 2)}
    no_under = rubric.get("no_shot_under_s", 0.9)
    inv_by = {s["shot"]: s for s in inventory.get("shots", [])}
    elements = []
    for i, s in enumerate(segs):
        sid = _seg_id(s, i)
        d = s["trim"]["out"] - s["trim"]["in"]
        deltas = []
        if d < no_under:
            deltas.append(f"too short ({d:.2f}s < {no_under}s)")
        # A generated segment has no inventory entry; its motion/energy is judged
        # subjectively on the frames, not by the source-footage heuristic.
        # NOTE: inv_by maps shot -> the whole shot object; motion lives under
        # `shot["features"]["motion"]`, not on the shot object itself. We must
        # reach into the features dict or every source shot is falsely flagged
        # as low-motion (feat.get("motion", 0) is always 0 on the shot object).
        feat = inv_by.get(sid, {}).get("features", {}) if not s.get("generated") else {}
        # Only apply the motion heuristic when features are actually present.
        # An un-enriched inventory has NO `features` key; treating missing motion
        # as 0 would falsely flag every source shot as dead-air. Skip the check
        # (treat as unknown) when features are absent or `motion` is not recorded.
        if (not s.get("generated")) and "motion" in feat and feat["motion"] < 1.0:
            deltas.append("low motion (may be dead air)")
        elements.append({"shot": sid,
                         "decision": "keep" if not deltas else "drop",
                         "why": "; ".join(deltas) if deltas else (s.get("why", "") or "passes rubric"),
                         "reward_delta": round(-0.2 * len(deltas), 2)})
    dense = sum(e["reward_delta"] for e in elements)
    overall = 0.5 * score_sig + 0.5 * dense
    if subjective is not None:
        overall = 0.6 * overall + 0.4 * subjective
    return {"overall_score": round(overall, 3), "metrics": metrics,
            "reward": round(overall + dense, 3), "elements": elements,
            "revise": [e["shot"] for e in elements if e["decision"] == "drop"][:8]}


def cmd_critic(schema: dict, inventory: dict, rubric: dict,
               render_dur: float | None = None, subjective: float | None = None,
               dataset: str | None = None, group_id: str | None = None,
               candidate: int | None = None, chosen_by: str = "critic") -> dict:
    """Score a schema, and optionally append the outcome to the training log.

    Logging a reward is how the render + vision leg enters GRPO: the group was
    written by `propose` (with the cheap objective reward), and this refines the
    selected candidate's reward to the one the real critique produced.
    """
    out = _score_objective(schema, inventory, rubric, subjective)
    if dataset and group_id:
        idx = int(candidate or 0)
        _log_jsonl(dataset, {
            "kind": "reward", "group_id": group_id, "candidate": idx,
            "chosen_by": chosen_by,
            "reward": out["reward"], "objective_reward": out["reward"],
            "overall_obj": out["overall_score"],
            "dense_obj": round(sum(e["reward_delta"] for e in out["elements"]), 4),
            "subjective": subjective, "logged_at": int(time.time()),
        })
        out["logged"] = {"dataset": dataset, "group_id": group_id, "candidate": idx,
                         "chosen_by": chosen_by, "reward": out["reward"]}
    return out


# --------------------------------------------------------------------------
# revise
# --------------------------------------------------------------------------
def cmd_revise(schema: dict, critic: dict) -> dict:
    drop = {e["shot"] for e in critic.get("elements", []) if e["decision"] == "drop"}
    kept = [s for i, s in enumerate(schema.get("structure", [])) if _seg_id(s, i) not in drop]
    if not kept:
        kept = schema.get("structure", [])
    t = 0.0
    for s in kept:
        d = s["trim"]["out"] - s["trim"]["in"]
        # The schema may be hand-authored or externally supplied and not carry
        # `timeline`; synthesize it rather than crashing on KeyError.
        s.setdefault("timeline", {})
        s["timeline"]["in"] = round(t, 3)
        t += d
        s["timeline"]["out"] = round(t, 3)
    schema["structure"] = kept
    schema["meta"]["revised_by"] = "critic"
    return schema


# --------------------------------------------------------------------------
# taste model: train / score / inspect
# --------------------------------------------------------------------------
def cmd_train(dataset: str, identity: str, style: str | None = None,
              creator: str = "default", epochs: int = 150, lr_muon: float = 0.005,
              lr_adamw: float = 0.005, temperature: float = 0.5, clip_eps: float = 0.2,
              entropy_coef: float = 0.01, pref_coef: float = 0.5, lambda_dense: float = 1.0,
              grad_clip: float = 1.0, inner_steps: int = 1, batch_groups: int = 16,
              revealed_pref_bonus: float = 1.0,
              d_h: int | None = None, d_z: int | None = None, seed: int = 0,
              holdout_every: int = 5, warm_start: bool = True) -> dict:
    """Fit the shared style-brain + this creator's latent, writing both GGUFs."""
    tm = _taste()
    kw: dict = {}
    if d_h:
        kw["d_h"] = int(d_h)
    if d_z:
        kw["d_z"] = int(d_z)
    return tm.train(dataset, identity, style=style, creator=creator, epochs=int(epochs),
                    lr_muon=lr_muon, lr_adamw=lr_adamw, temperature=temperature,
                    clip_eps=clip_eps, entropy_coef=entropy_coef, pref_coef=pref_coef,
                    lambda_dense=lambda_dense, grad_clip=grad_clip,
                    inner_steps=int(inner_steps), batch_groups=int(batch_groups),
                    revealed_pref_bonus=revealed_pref_bonus,
                    seed=int(seed),
                    holdout_every=int(holdout_every),
                    warm_start=warm_start, **kw)


def cmd_taste(identity: str, style: str | None = None,
              features: str | None = None, schema: str | None = None,
              inventory: str | None = None, rubric: str | None = None) -> dict:
    """Score one or more candidate feature dicts with a trained identity."""
    tm = _taste()
    scorer = tm.TasteScorer.load(identity, style)
    if features:
        payload = _fixture(features)
        if isinstance(payload, list):
            return {"scores": scorer.score_many(payload), "feature_spec": scorer.feature_spec,
                    "n": len(payload)}
        return {"score": scorer.score_features(payload), "feature_spec": scorer.feature_spec}
    if not (schema and inventory and rubric):
        raise RuntimeError("taste needs --features or all of --schema/--inventory/--rubric")
    feats = tm.features_for(scorer.feature_spec, _fixture(schema), _fixture(inventory),
                            _fixture(rubric))
    return {"score": scorer.score_features(feats), "features": feats,
            "feature_spec": scorer.feature_spec}


def cmd_taste_features(schema: str, inventory: str, rubric: str,
                       spec: str | None = None) -> dict:
    """The ordered feature vector a schema maps to (layout is in the artifact)."""
    tm = _taste()
    use = spec or tm.FEATURE_SPEC_VIDEO
    feats = tm.features_for(use, _fixture(schema), _fixture(inventory), _fixture(rubric))
    return {"feature_spec": use, "features": feats, "vector": tm.to_vector(feats, use),
            "names": tm.FEATURE_SPECS[use]}


def cmd_identity_init(identity: str, style: str | None = None, creator: str = "default",
                      spec: str | None = None) -> dict:
    """Create a zero-latent identity so a new creator can run the loop at once."""
    tm = _taste()
    return tm.init_identity(identity, style=style, creator=creator,
                            spec=spec or tm.FEATURE_SPEC_VIDEO)


def cmd_identity_info(identity: str, style: str | None = None) -> dict:
    return _taste().TasteScorer.load(identity, style).info()


def cmd_taste_status(identity: str | None = None, dataset: str | None = None,
                     style: str | None = None) -> dict:
    """One-shot status of the wired-in taste loop: identity + training data."""
    tm = _taste()
    out: dict = {"identity": None, "dataset": None}
    if identity:
        if os.path.exists(identity):
            out["identity"] = {**tm.TasteScorer.load(identity, style).info(), "exists": True}
        else:
            out["identity"] = {"path": identity, "exists": False,
                               "hint": "run identity_init (or train) to create it"}
    if dataset:
        out["dataset"] = tm.dataset_status(dataset)
    return out


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
def temp_dir() -> str:
    import tempfile
    return tempfile.mkdtemp(prefix="dshe_")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inventory")
    p.add_argument("src")
    p.add_argument("--threshold", type=float, default=27.0)
    p.add_argument("--min-scene-len", type=float, default=0.6)

    p = sub.add_parser("features")
    p.add_argument("src"); p.add_argument("inventory")

    p = sub.add_parser("propose")
    p.add_argument("inventory"); p.add_argument("rubric")
    p.add_argument("--group", type=int, default=int(os.getenv("DSH_EDITAPART_GROUP", "1")),
                   help="number of candidate schemas in the group (1 = legacy single schema)")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"))
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))
    p.add_argument("--dataset", default=os.getenv("DSH_EDITAPART_DATASET"))
    p.add_argument("--clip-id", default=None)
    p.add_argument("--select", choices=["auto", "taste", "objective"], default="auto")
    p.add_argument("--no-log", action="store_true", help="do not append the group to the dataset")

    p = sub.add_parser("render")
    p.add_argument("src"); p.add_argument("schema"); p.add_argument("out")

    p = sub.add_parser("review_frames")
    p.add_argument("src"); p.add_argument("schema"); p.add_argument("outdir")
    p.add_argument("--size", type=int, default=640)

    p = sub.add_parser("critic")
    p.add_argument("schema"); p.add_argument("inventory"); p.add_argument("rubric")
    p.add_argument("--render-dur", type=float, default=None)
    p.add_argument("--subjective", type=float, default=None)
    p.add_argument("--dataset", default=os.getenv("DSH_EDITAPART_DATASET"))
    p.add_argument("--group-id", default=None)
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--chosen-by", choices=["critic", "creator", "agent", "human"],
                   default="critic",
                   help="who picked this candidate: a creator/agent pick is logged "
                        "as a revealed preference")

    p = sub.add_parser("revise")
    p.add_argument("schema"); p.add_argument("critic")

    p = sub.add_parser("train")
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
    p.add_argument("--d-h", type=int, default=None)
    p.add_argument("--d-z", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--batch-groups", type=int, default=16)
    p.add_argument("--revealed-pref-bonus", type=float, default=1.0)
    p.add_argument("--holdout-every", type=int, default=5)
    p.add_argument("--no-warm-start", action="store_true")

    p = sub.add_parser("taste")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"), required=False)
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))
    p.add_argument("--features", default=None)
    p.add_argument("--schema", default=None)
    p.add_argument("--inventory", default=None)
    p.add_argument("--rubric", default=None)

    p = sub.add_parser("taste_features")
    p.add_argument("--schema", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--rubric", required=True)
    p.add_argument("--spec", default=None)

    p = sub.add_parser("taste_status")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"))
    p.add_argument("--dataset", default=os.getenv("DSH_EDITAPART_DATASET"))
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))

    p = sub.add_parser("identity_init")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"), required=False)
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))
    p.add_argument("--creator", default=os.getenv("DSH_EDITAPART_CREATOR", "default"))
    p.add_argument("--spec", default=None)

    p = sub.add_parser("identity_info")
    p.add_argument("--identity", default=os.getenv("DSH_EDITAPART_IDENTITY"), required=False)
    p.add_argument("--style", default=os.getenv("DSH_EDITAPART_STYLE"))

    args = ap.parse_args()
    try:
        if args.cmd == "inventory":
            result = cmd_inventory(args.src, args.threshold, args.min_scene_len)
        elif args.cmd == "features":
            result = cmd_features(args.src, json.loads(args.inventory))
        elif args.cmd == "propose":
            result = cmd_propose(json.loads(args.inventory), json.loads(args.rubric),
                                 group=args.group, identity=args.identity, style=args.style,
                                 dataset=args.dataset, clip_id=args.clip_id,
                                 select=args.select, log=not args.no_log)
        elif args.cmd == "render":
            result = cmd_render(args.src, json.loads(args.schema), args.out)
        elif args.cmd == "review_frames":
            result = cmd_review_frames(args.src, json.loads(args.schema), args.outdir, args.size)
        elif args.cmd == "critic":
            result = cmd_critic(json.loads(args.schema), json.loads(args.inventory),
                                json.loads(args.rubric), args.render_dur, args.subjective,
                                dataset=args.dataset, group_id=args.group_id,
                                candidate=args.candidate, chosen_by=args.chosen_by)
        elif args.cmd == "revise":
            result = cmd_revise(json.loads(args.schema), json.loads(args.critic))
        elif args.cmd == "train":
            if not args.dataset or not args.identity:
                raise RuntimeError("train needs --dataset and --identity "
                                   "(or DSH_EDITAPART_DATASET / DSH_EDITAPART_IDENTITY)")
            result = cmd_train(args.dataset, args.identity, style=args.style,
                               creator=args.creator, epochs=args.epochs, lr_muon=args.lr_muon,
                               lr_adamw=args.lr_adamw, temperature=args.temperature,
                               clip_eps=args.clip_eps, entropy_coef=args.entropy_coef,
                               pref_coef=args.pref_coef, lambda_dense=args.lambda_dense,
                               grad_clip=args.grad_clip, inner_steps=args.inner_steps,
                               batch_groups=args.batch_groups,
                               revealed_pref_bonus=args.revealed_pref_bonus,
                               d_h=args.d_h, d_z=args.d_z, seed=args.seed,
                               holdout_every=args.holdout_every,
                               warm_start=not args.no_warm_start)
        elif args.cmd == "taste":
            if not args.identity:
                raise RuntimeError("taste needs --identity (or DSH_EDITAPART_IDENTITY)")
            result = cmd_taste(args.identity, style=args.style, features=args.features,
                               schema=args.schema, inventory=args.inventory, rubric=args.rubric)
        elif args.cmd == "taste_features":
            result = cmd_taste_features(args.schema, args.inventory, args.rubric, args.spec)
        elif args.cmd == "taste_status":
            result = cmd_taste_status(args.identity, args.dataset, args.style)
        elif args.cmd == "identity_init":
            if not args.identity:
                raise RuntimeError("identity_init needs --identity (or DSH_EDITAPART_IDENTITY)")
            result = cmd_identity_init(args.identity, style=args.style, creator=args.creator,
                                       spec=args.spec)
        elif args.cmd == "identity_info":
            if not args.identity:
                raise RuntimeError("identity_info needs --identity (or DSH_EDITAPART_IDENTITY)")
            result = cmd_identity_info(args.identity, style=args.style)
        else:
            ap.error(f"unknown command {args.cmd}")
    except Exception as exc:  # noqa: BLE001 — surfaced to the plugin as a tool error
        print(json.dumps({"error": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
