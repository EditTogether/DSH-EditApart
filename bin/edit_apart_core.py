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
import copy
import hashlib
import json
import math
import os
import re
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


def _num(value, field: str, lo: float | None = None, hi: float | None = None) -> float:
    """Validate one schema value that is interpolated into the ffmpeg filtergraph.

    A filtergraph cannot execute commands, but it CAN be broken or side-channelled
    by a value carrying `,`/`;`/`=` or a filter reference (`movie=`, `zmq`,
    `sendcmd`). Schemas here are model-authored, so a malformed field used to
    either crash the render with an opaque ffmpeg error or splice arbitrary filter
    text into the graph. Every value that reaches the graph goes through here.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"schema field {field!r} must be a number, got {value!r}")
    if not math.isfinite(f):
        raise RuntimeError(f"schema field {field!r} must be a finite number, got {value!r}")
    if lo is not None and f < lo:
        raise RuntimeError(f"schema field {field!r}={f} is below the minimum {lo}")
    if hi is not None and f > hi:
        raise RuntimeError(f"schema field {field!r}={f} is above the maximum {hi}")
    return f


_FF_CROP_RE = re.compile(r"^-?\d{1,5}:-?\d{1,5}:-?\d{1,5}:-?\d{1,5}$")


def _ff_crop(value) -> str:
    """A crop geometry string, validated (w:h:x:y, integers only)."""
    text = str(value)
    if not _FF_CROP_RE.match(text):
        raise RuntimeError(
            f"schema field 'transform.crop' must be 'w:h:x:y' with integers, got {value!r}")
    return text


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
    try:
        return _inventory_from(src, threshold, min_scene_len, out_dir)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _inventory_from(src: str, threshold: float, min_scene_len: float, out_dir: str) -> dict:
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


#: Dataset record version. v3 carries what a REPLAY needs — the constraints, the
#: evidence, the timelines and the source identity — not only the derived numbers.
#: Without them a logged group cannot be re-derived, so a defect in the feature
#: extractor or a change to the critic would make the whole corpus un-remeasurable
#: and force a re-shoot. Records stay readable across versions: the trainer reads
#: the fields it knows and ignores the rest, and `replay` reports older records as
#: not replayable rather than guessing.
DATASET_SCHEMA_VERSION = 3
ENGINE_VERSION = "edit_apart_core/0.2.1"     # bump when the extractor changes


def _canonical_hash(obj) -> str:
    """Deterministic content hash of a JSON-able object.

    `sort_keys` + fixed separators so the hash depends on the CONTENT, not on dict
    ordering or whitespace: two runs that produce the same group produce the same
    hash, which is what lets a ledger key annotations to a revision.
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_file(path: str | None) -> str | None:
    """Streaming content hash of a source file, or None when it is unavailable.

    None is recorded as *no hash*, never as a wrong one: a replay must be able to
    tell "the source is not identified" from "the source changed".
    """
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


# ── effect parameters (the "MIMO" effects) ──────────────────────────────────
# A transition joins TWO clips, ducking relates speech to music, an overlay
# composites layers. The selection grid above varies none of them, so a taste
# model over the v1 layout can only choose WHICH selection to use — it cannot set
# how the inputs combine. These are the axes such an effect actually has, and
# `_refine_effects` searches them with the identity's score.
TRANSITION_TYPES = ("cut", "dissolve")
TRANSITION_ALIGNMENTS = ("center", "start", "end")
TRANSITION_DURS = (0.0, 0.2, 0.4, 0.8, 1.2)
DUCK_DEPTHS = (0.0, 0.3, 0.6)
GRADE_STRENGTHS = (0.0, 1.0, 2.0)

EFFECT_DEFAULTS = {"transition": "cut", "transition_dur": 0.0,
                   "transition_align": "center", "duck": 0.0, "grade": 0.0}

#: Candidate grid for effect-spanning groups (`--effect-grid`). The default grid
#: varies shot SELECTION only, so its candidates are identical in every effect
#: feature — an identity trained on those groups can never learn an effect
#: preference, and `refine` would have nothing to apply. These entries are paired
#: with the selection grid so one group spans both, which is what lets an external
#: pick between two effect settings become a revealed preference.
EFFECT_CANDIDATE_GRID = (
    {},                                                                    # rubric verbatim
    {"transition": "dissolve", "transition_dur": 0.4, "transition_align": "center"},
    {"transition": "dissolve", "transition_dur": 0.8, "transition_align": "center"},
    {"transition": "dissolve", "transition_dur": 0.8, "transition_align": "start"},
    {"duck": 0.3},
    {"duck": 0.6, "grade": 1.0},
)

#: The axes swept by the refiner, in a FIXED order so a run is reproducible.
#: Each option is a knob OVERRIDE SET rather than a single value, because the
#: parameters are coupled: a transition needs a type AND a positive duration, so
#: sweeping "type" and "duration" separately gets stuck the moment the type changes
#: while the duration is still 0 (the renderer then has no transition to score).
EFFECT_AXES = (
    ("transition", tuple({"transition": t, "transition_dur": d}
                         for t, d in (("cut", 0.0), ("dissolve", 0.2), ("dissolve", 0.4),
                                      ("dissolve", 0.8), ("dissolve", 1.2)))),
    ("transition_align", tuple({"transition_align": a} for a in TRANSITION_ALIGNMENTS)),
    ("duck", tuple({"duck": d} for d in DUCK_DEPTHS)),
    ("grade", tuple({"grade": g} for g in GRADE_STRENGTHS)),
)


def apply_effects(schema: dict, knobs: dict) -> dict:
    """Write multi-input effect parameters onto every segment of a schema.

    `cut` clears the duration and the alignment; any other type carries both,
    because the renderer's implicit choice (each side takes half the duration) is
    a DECISION that used to be made silently and never recorded. Grade strength
    scales saturation/contrast from the rubric's own baseline, so grading never
    contradicts the rubric's intent — it interpolates around it.
    """
    type_ = str(knobs.get("transition", "cut"))
    dur = float(knobs.get("transition_dur", 0.0) or 0.0)
    align = str(knobs.get("transition_align", "center"))
    duck = float(knobs.get("duck", 0.0) or 0.0)
    grade = float(knobs.get("grade", 0.0) or 0.0)
    for seg in schema.get("structure") or []:
        if type_ == "cut" or dur <= 0.0:
            seg["transition"] = {"type": "cut", "dur": 0.0, "params": {}}
        else:
            seg["transition"] = {"type": type_, "dur": round(dur, 3),
                                 "params": {"alignment": align}}
        if duck:
            seg.setdefault("audio", {})["duck"] = round(duck, 4)
        if grade:
            eq = seg.setdefault("grade", {}).setdefault("eq", {})
            eq["saturation"] = round(1.0 + 0.10 * grade, 4)
            eq["contrast"] = round(0.05 * grade, 4)
    return schema


def knobs_from_schema(schema: dict) -> dict:
    """Recover the effect knobs a schema already carries (so a refinement starts
    from the candidate under test rather than from a hardcoded default)."""
    knobs = dict(EFFECT_DEFAULTS)
    for seg in schema.get("structure") or []:
        tr = seg.get("transition") or {}
        if tr.get("type") not in (None, "", "cut") and float(tr.get("dur") or 0.0) > 0:
            knobs["transition"] = str(tr["type"])
            knobs["transition_dur"] = float(tr["dur"])
            knobs["transition_align"] = str((tr.get("params") or {}).get("alignment", "center"))
        audio = seg.get("audio") or {}
        if float(audio.get("duck") or 0.0) > 0:
            knobs["duck"] = float(audio["duck"])
        eq = (seg.get("grade") or {}).get("eq") or {}
        sat = float(eq.get("saturation", 1.0) or 1.0)
        if abs(sat - 1.0) > 1e-9:
            knobs["grade"] = round((sat - 1.0) / 0.10, 3)
        break
    return knobs


def _effect_score(tm, spec: str, schema: dict, inventory: dict, rubric: dict,
                  scorer) -> float:
    if scorer is None:
        return 0.0
    return float(scorer.score_features(tm.features_for(spec, schema, inventory, rubric)))


def refine_effects(tm, schema: dict, inventory: dict, rubric: dict, scorer,
                   steps: int = 3, spec: str | None = None) -> dict:
    """Coordinate ascent over the effect parameters, scored by the identity.

    This is what "apply the taste to a multi-input effect" MEANS operationally: the
    identity does not only rank a fixed list of whole candidates, it chooses the
    parameter values. Deterministic — the axes are swept in a fixed order, a tie
    keeps the incumbent, and `steps` bounds the passes — so two runs with the same
    identity and the same starting schema agree exactly.

    A `scorer` of None returns the schema untouched: with no identity there is no
    taste to apply, and inventing one (e.g. maximising the objective critic here)
    would silently turn a rubric rule into a taste claim.
    """
    spec = spec or tm.FEATURE_SPEC_VIDEO_V2
    trace: list[dict] = []
    if scorer is None:
        return {"schema": schema, "knobs": knobs_from_schema(schema),
                "trace": trace, "score": None}
    best = copy.deepcopy(schema)
    knobs = knobs_from_schema(best)
    best_score = _effect_score(tm, spec, best, inventory, rubric, scorer)
    for _ in range(max(0, int(steps))):
        improved = False
        for axis, options in EFFECT_AXES:
            for option in options:
                if all(knobs.get(k) == v for k, v in option.items()):
                    continue                      # already there: nothing to try
                trial_knobs = {**knobs, **option}
                trial = apply_effects(copy.deepcopy(best), trial_knobs)
                score = _effect_score(tm, spec, trial, inventory, rubric, scorer)
                if score > best_score + 1e-12:
                    best, best_score, knobs = trial, score, trial_knobs
                    trace.append({"axis": axis, "option": dict(option),
                                  "score": round(score, 6)})
                    improved = True
        if not improved:
            break
    return {"schema": best, "knobs": knobs, "trace": trace, "score": round(best_score, 6)}


def cmd_replay(dataset: str, limit: int | None = None) -> dict:
    """Re-derive every logged group and check it against what the record stored.

    This is what turns "we keep what a replay needs" from a policy into a property.
    For each v3 group the record itself supplies the timeline, the inventory, the
    rubric and the feature layout, and this recomputes:

      * each candidate's feature vector — compared with the vector written at log time;
      * the objective reward (and its components) from the same three inputs;
      * the recorded content / rubric / inventory hashes, plus the source media hash
        when one was recorded and the file is still on disk.

    The SUBJECTIVE leg is a person's or agent's judgment, so it is reported as not
    recomputable and never re-invented. Older records (no timeline/evidence, or
    `dataset_schema_version` < 3) are counted as NOT replayable — an explicit state,
    not a silent skip and not a failure.
    """
    tm = _taste()
    report = {"dataset": dataset, "groups": 0, "replayable": 0, "not_replayable": 0,
              "clean": 0, "candidates_checked": 0, "features_match": 0,
              "rewards_match": 0, "source_hash_checked": 0, "source_hash_unavailable": 0,
              "unparsable_lines": 0, "mismatches": []}
    if not os.path.isfile(dataset):
        raise FileNotFoundError(f"dataset not found: {dataset}")
    with open(dataset, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                report["unparsable_lines"] += 1
                continue
            if rec.get("kind") != "group":
                continue
            if limit is not None and report["groups"] >= int(limit):
                break
            report["groups"] += 1
            spec = rec.get("feature_spec")
            rubric, inventory = rec.get("rubric"), rec.get("inventory")
            cands = rec.get("candidates") or []
            structural = (int(rec.get("dataset_schema_version") or 0) >= DATASET_SCHEMA_VERSION
                          and isinstance(rubric, dict) and isinstance(inventory, dict)
                          and bool(cands)
                          and all(isinstance(c.get("schema"), dict) for c in cands)
                          and spec in tm.FEATURE_SPECS)
            if not structural:
                report["not_replayable"] += 1
                continue
            report["replayable"] += 1
            problems: list[str] = []
            if rec.get("rubric_hash") != _canonical_hash(rubric):
                problems.append("rubric_hash")
            if rec.get("inventory_hash") != _canonical_hash(inventory):
                problems.append("inventory_hash")
            if rec.get("content_hash") != _canonical_hash({
                    "feature_spec": spec, "rubric": rubric, "inventory": inventory,
                    "candidates": [c["schema"] for c in cands]}):
                problems.append("content_hash")
            recorded_sha = rec.get("source_sha256")
            if recorded_sha:
                actual_sha = _sha256_file(inventory.get("source"))
                if actual_sha is None:
                    report["source_hash_unavailable"] += 1
                elif actual_sha != recorded_sha:
                    problems.append("source_sha256")
                else:
                    report["source_hash_checked"] += 1
            for c in cands:
                report["candidates_checked"] += 1
                idx = c.get("idx")
                derived = tm.features_for(spec, c["schema"], inventory, rubric)
                stored = c.get("features") or {}
                drift = {k: [stored.get(k), derived.get(k)] for k in set(stored) | set(derived)
                         if abs(float(stored.get(k, 0.0) or 0.0)
                                - float(derived.get(k, 0.0) or 0.0)) > 1e-9}
                if drift:
                    problems.append(f"features[{idx}]")
                else:
                    report["features_match"] += 1
                crit = _score_objective(c["schema"], inventory, rubric, None)
                reward_ok = (abs(float(crit["reward"]) - float(c.get("reward_obj") or 0.0)) <= 1e-9
                             and abs(float(crit["overall_score"])
                                     - float(c.get("overall_obj") or 0.0)) <= 1e-9
                             and abs(sum(e["reward_delta"] for e in crit["elements"])
                                     - float(c.get("dense_obj") or 0.0)) <= 1e-4)
                if reward_ok:
                    report["rewards_match"] += 1
                else:
                    problems.append(f"reward[{idx}]")
            if problems:
                report["mismatches"].append({"group_id": rec.get("group_id"),
                                             "content_hash": rec.get("content_hash"),
                                             "problems": sorted(set(problems))})
            else:
                report["clean"] += 1
    report["ok"] = bool(report["replayable"] and not report["mismatches"]
                        and not report["not_replayable"])
    return report


def _grid_for(group: int) -> list:
    """The deterministic candidate grid, capped at the table size."""
    n = max(1, min(int(group), len(CANDIDATE_GRID)))
    return CANDIDATE_GRID[:n]


def cmd_propose(inventory: dict, rubric: dict, group: int = 1, identity: str | None = None,
                style: str | None = None, dataset: str | None = None,
                clip_id: str | None = None, select: str = "auto",
                log: bool = True, refine: int = 0,
                feature_spec: str | None = None, effect_grid: bool = False) -> dict:
    """Propose a group of candidate schemas, select one, and log the group.

    Selection: with an identity, `select=auto|taste` picks the taste-model
    argmax; otherwise (or with `select=objective`) it picks the best objective
    reward. Either way `group=1` selects the only candidate, which is the legacy
    schema.

    `refine=N` additionally lets the identity choose the MULTI-INPUT EFFECT
    parameters (transition type/duration/alignment, ducking, grade strength) by
    coordinate ascent, and appends the tuned schema to the group as one more
    candidate. It requires an identity whose trunk was trained on the effect-aware
    layout (`video/v2`): a v1 trunk has 16 inputs and cannot score a 26-dim vector,
    so the request is declined with a reason rather than silently ignored.
    """
    tm = _taste()
    grid = _grid_for(group)
    clip = clip_id or str(inventory.get("source") or "")
    cands = []
    for i, (target_factor, min_factor, skip_short, keep_max) in enumerate(grid):
        schema = _propose_variant(inventory, rubric, target_factor, min_factor,
                                  skip_short, keep_max)
        effect_knobs = {}
        if effect_grid:
            effect_knobs = dict(EFFECT_CANDIDATE_GRID[i % len(EFFECT_CANDIDATE_GRID)])
            if effect_knobs:
                apply_effects(schema, effect_knobs)
        crit = _score_objective(schema, inventory, rubric, None)
        cands.append({
            "idx": i, "schema": schema, "critic": crit,
            "knobs": {"target_factor": target_factor, "min_shot_factor": min_factor,
                      "skip_short": skip_short, "keep_max": keep_max, **effect_knobs},
        })

    scorer = None
    spec = None
    if identity:
        scorer = tm.TasteScorer.load(identity, style)
        spec = scorer.feature_spec
    if spec is None:
        # An effect-spanning group is only meaningful under a layout that can SEE
        # the effects, so it selects the effect-aware one unless told otherwise.
        spec = feature_spec or (tm.FEATURE_SPEC_VIDEO_V2 if effect_grid
                                else tm.FEATURE_SPEC_VIDEO)
    feats = [tm.features_for(spec, c["schema"], inventory, rubric) for c in cands]
    scores = scorer.score_many(feats) if scorer else [None] * len(cands)

    # ── taste applied to the multi-input effects ─────────────────────────────
    refined: dict | None = None
    refine_note: str | None = None
    if refine > 0:
        if scorer is None:
            refine_note = "no identity: there is no taste to apply to the effects"
        elif spec != tm.FEATURE_SPEC_VIDEO_V2:
            refine_note = (f"identity trunk is trained on {spec}, which has no effect "
                           "features; retrain on the effect-aware layout to tune effects")
        else:
            seed = max(range(len(cands)), key=lambda i: scores[i])
            refined = refine_effects(tm, cands[seed]["schema"], inventory, rubric,
                                     scorer, steps=refine, spec=spec)
            if refined["score"] is not None and refined["score"] > scores[seed] + 1e-12:
                feats.append(tm.features_for(spec, refined["schema"], inventory, rubric))
                scores.append(refined["score"])
                cands.append({
                    "idx": len(cands), "schema": refined["schema"],
                    "critic": _score_objective(refined["schema"], inventory, rubric, None),
                    "knobs": {**cands[seed]["knobs"], "effects": refined["knobs"],
                              "refined_from": seed},
                })
            else:
                refine_note = "the identity scored no effect change above the seed candidate"

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
        # Everything a replay needs, in the record itself: the constraint set that
        # made these alternatives comparable, the evidence the features came from,
        # the timelines (not just knobs and vectors), and the source identity.
        rubric_hash = _canonical_hash(rubric)
        inventory_hash = _canonical_hash(inventory)
        source_sha256 = _sha256_file(inventory.get("source"))
        content_hash = _canonical_hash({
            "feature_spec": spec, "rubric": rubric, "inventory": inventory,
            "candidates": [c["schema"] for c in cands],
        })
        _log_jsonl(dataset, {
            "kind": "group", "group_id": gid, "clip_id": clip,
            "dataset_schema_version": DATASET_SCHEMA_VERSION, "engine": ENGINE_VERSION,
            "feature_spec": spec, "created": int(time.time()),
            "select_by": select_by, "selected": selected,
            "content_hash": content_hash, "rubric_hash": rubric_hash,
            "inventory_hash": inventory_hash, "source_sha256": source_sha256,
            "rubric": rubric, "inventory": inventory,
            "candidates": [{
                "idx": c["idx"], "knobs": c["knobs"], "features": feats[c["idx"]],
                "reward_obj": c["critic"]["reward"], "taste_score": scores[c["idx"]],
                "overall_obj": c["critic"]["overall_score"],
                "dense_obj": round(sum(e["reward_delta"] for e in c["critic"]["elements"]), 4),
                "n_segments": len(c["schema"]["structure"]),
                "schema": c["schema"],
            } for c in cands],
        })

    chosen = cands[selected]
    out = {"meta": dict(chosen["schema"]["meta"]), "structure": chosen["schema"]["structure"],
           "globals": chosen["schema"]["globals"]}
    out["group"] = {
        "group_id": gid, "feature_spec": spec, "size": len(cands),
        "selected": selected, "select_by": select_by, "dataset": dataset,
        "refine": refine, "refine_note": refine_note,
        "effects": (refined["knobs"] if refined else None),
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
            # The candidate's OWN schema, so a caller can render (and critique) a
            # different candidate than the one selected — the workflow the skill
            # asks for when the agent or creator overrides the selection. Without
            # it, a caller critiquing "the picked candidate" can only pass the
            # selected schema and silently mislabel the reward.
            "schema": c["schema"],
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


def _has_audio(src: str) -> bool:
    """Whether an input carries an audio stream at all.

    The renderer used to map `[i:a]` unconditionally, so ONE input without audio
    (a muted screen recording, a GIF-sourced clip, a camera dump) failed the whole
    ffmpeg invocation with "Stream specifier 'a' matched no streams".
    """
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0", src],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


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
    used_inputs: set[int] = set()
    for i, seg in enumerate(segs):
        s = src_index(seg)
        used_inputs.add(s)
        trim = seg.get("trim") or {}
        if "in" not in trim or "out" not in trim:
            raise RuntimeError(f"segment {i} has no trim {{in,out}} — the renderer needs both")
        t_in = _num(trim["in"], f"structure[{i}].trim.in", 0.0, 10 ** 6)
        t_out = _num(trim["out"], f"structure[{i}].trim.out", 0.0, 10 ** 6)
        if t_out <= t_in:
            raise RuntimeError(f"segment {i} has trim.out ({t_out}) <= trim.in ({t_in})")
        parts = ["trim=" + f"start={t_in:g}:end={t_out:g}", "setpts=PTS-STARTPTS"]
        retime = seg.get("retime", 0.0)
        if retime not in (0.0, None):
            parts.append(f"setpts=PTS/{_num(retime, f'structure[{i}].retime', 0.01, 100.0):g}")
        tf = seg.get("transform") or {}
        scale = _num(tf.get("scale", 1.0), f"structure[{i}].transform.scale", 0.01, 20.0)
        if scale != 1.0:
            parts.append(f"scale=iw*{scale:g}:ih*{scale:g}")
        crop = tf.get("crop")
        if crop:
            parts.append(f"crop={_ff_crop(crop)}")
        grade = seg.get("grade") or {}
        eq = grade.get("eq") or {}
        eqopts = [f"{k}={_num(eq[k], f'structure[{i}].grade.eq.{k}'):g}"
                  for k in ("brightness", "contrast", "saturation", "gamma") if k in eq]
        if eqopts:
            parts.append("eq=" + ":".join(eqopts))
        v_filter.append(f"[{s}:v]" + ",".join(parts) + f",format=yuv420p[v{i}]")
        in_labels.append(f"[v{i}]")
    concat_v = "".join(in_labels) + f"concat=n={len(segs)}:v=1:a=0[outv]"

    # Audio presence must be probed AFTER generated assets are registered above,
    # and only for the inputs the segments actually reference.
    audio_ok = {i: _has_audio(inputs[i]) for i in sorted(used_inputs)}
    any_audio = any(audio_ok.values())
    a_filter: list[str] = []
    a_labels: list[str] = []
    if any_audio:
        for i, seg in enumerate(segs):
            s = src_index(seg)
            trim = seg.get("trim") or {}
            t_in = _num(trim["in"], f"structure[{i}].trim.in", 0.0, 10 ** 6)
            t_out = _num(trim["out"], f"structure[{i}].trim.out", 0.0, 10 ** 6)
            adur = t_out - t_in
            if audio_ok.get(s):
                a = f"[{s}:a]" + f"atrim=start={t_in:g}:end={t_out:g},asetpts=PTS-STARTPTS"
            else:
                # No audio on THIS input (muted screen recording, GIF-sourced clip,
                # camera dump): synthesize silence for the segment rather than
                # failing the whole render on "Stream specifier 'a' matched no streams".
                a = (f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                     f"atrim=start=0:end={max(0.05, adur):g},asetpts=PTS-STARTPTS")
            au = seg.get("audio") or {}
            lvl = _num(au.get("level", 1.0), f"structure[{i}].audio.level", 0.0, 100.0)
            if lvl != 1.0:
                a += f",volume={lvl:g}"
            fade = _num(au.get("fade", 0.0), f"structure[{i}].audio.fade", 0.0, 10 ** 6)
            if fade:
                a += f",afade=t=in:st=0:d={fade:g},afade=t=out:st={max(0, adur - fade):g}:d={fade:g}"
            a += f"[a{i}]"
            a_filter.append(a)
            a_labels.append(f"[a{i}]")
        a_chain = "".join(a_labels) + f"concat=n={len(a_labels)}:v=0:a=1,loudnorm[aout]"
    else:
        # NO input carries audio at all. Emit one silent track of the total length
        # and deliberately skip loudnorm: single-pass loudnorm on digital silence
        # divides by zero energy and hands the AAC encoder NaN, which failed the
        # render with "Input contains (near) NaN/+-Inf".
        total = 0.0
        for i, seg in enumerate(segs):
            trim = seg.get("trim") or {}
            total += (_num(trim["out"], f"structure[{i}].trim.out", 0.0, 10 ** 6)
                      - _num(trim["in"], f"structure[{i}].trim.in", 0.0, 10 ** 6))
        a_chain = (f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                   f"atrim=start=0:end={max(0.05, total):g},asetpts=PTS-STARTPTS[aout]")
    fc = ";".join(v_filter + [concat_v]) + ";" + ";".join(a_filter + [a_chain])

    input_args: list[str] = []
    for inp in inputs:
        input_args += ["-i", inp]
    cmd = [FFMPEG, "-y", *input_args, "-filter_complex", fc,
           "-map", "[outv]", "-map", "[aout]", "-c:v", "libx264", "-preset",
           "veryfast", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # The BANNER is the first ~400 chars; the actual error is the last line.
        tail = "\n".join((r.stderr or "").strip().splitlines()[-3:]) or "(no stderr)"
        raise RuntimeError(f"ffmpeg render failed: {tail}")
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
    schema.setdefault("meta", {})["revised_by"] = "critic"
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
# selftest (environment conformance)
# --------------------------------------------------------------------------
def cmd_selftest() -> dict:
    """Prove this install can run the video loop end to end on a fixture.

    Guards the class of defect where a mis-resolved toolchain silently degrades a
    feature axis instead of failing: inventory must find shots, features must be
    non-degenerate, and a one-segment schema must render to a real file.
    """
    import tempfile
    d = tempfile.mkdtemp(prefix="dshe_selftest_")
    try:
        clip = os.path.join(d, "fixture.mp4")
        r = subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                            "testsrc=size=320x240:rate=10", "-t", "3",
                            "-pix_fmt", "yuv420p", clip], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg cannot synthesize a fixture: {r.stderr.strip()[-200:]}")
        inv = cmd_inventory(clip, 27.0, 0.6)
        if not inv["shots"]:
            raise RuntimeError("inventory found no shots in a synthetic clip")
        inv = cmd_features(clip, inv)
        feats = [s.get("features", {}) for s in inv["shots"]]
        motions = [f.get("motion", 0.0) for f in feats]
        if not any(m > 0 for m in motions):
            raise RuntimeError("every shot reports zero motion — features are degenerate")
        schema = {"meta": {"source": clip}, "structure": [{
            "shot": inv["shots"][0]["shot"],
            "trim": {"in": 0.0, "out": 1.0}, "retime": 0.0,
            "timeline": {"in": 0.0, "out": 1.0},
            "transition": {"type": "cut", "dur": 0.0, "params": {}},
            "transform": {"scale": 1.0, "crop": None, "position": None},
            "grade": {"lut": None, "eq": {}},
            "audio": {"level": 1.0, "duck": 0.0, "fade": 0.0}, "overlay": [], "why": "selftest"}],
            "globals": {"color": None, "audioMix": {}, "music": {"bed": None, "syncToBeat": False},
                        "pace": None}}
        out = os.path.join(d, "rendered.mp4")
        rendered = cmd_render(clip, schema, out)
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            raise RuntimeError("the renderer produced no output file")
        return {"ok": True, "ffmpeg": FFMPEG, "ffprobe": FFPROBE,
                "scenedetect": VENV_SCENEDETECT, "shots": len(inv["shots"]),
                "mean_motion": round(sum(motions) / len(motions), 3),
                "rendered_duration": rendered.get("duration")}
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
    p.add_argument("--refine", type=int, default=0, metavar="N",
                   help="let the identity choose the MULTI-INPUT EFFECT parameters "
                        "(transition type/duration/alignment, duck, grade) by coordinate "
                        "ascent; needs an identity whose trunk was trained on video/v2")
    p.add_argument("--feature-spec", choices=["video/v1", "video/v2"], default=None,
                   help="layout to LOG this group under (default: the identity's own, "
                        "else video/v1). video/v2 adds the effect-parameter features a "
                        "MIMO-effect taste model needs.")
    p.add_argument("--effect-grid", action="store_true",
                   help="span the MULTI-INPUT EFFECT parameters across the group "
                        "(transitions, ducking, grade) so a revealed pick can teach an "
                        "effect preference; logs under video/v2 by default")
    p.add_argument("--no-log", action="store_true", help="do not append the group to the dataset")

    p = sub.add_parser("replay", help="re-derive logged groups and check the record")
    p.add_argument("dataset")
    p.add_argument("--limit", type=int, default=0, help="check at most N groups")

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
    p.add_argument("--chosen-by", choices=["critic", "creator", "agent", "human", "taste"],
                   default="critic",
                   help="who picked this candidate. `creator`/`agent`/`human` is an "
                        "EXTERNAL judgment and is logged as a REVEALED preference; `taste` "
                        "means the per-creator identity selected it and no external judge "
                        "overrode — that is the system's own choice and teaches nothing "
                        "about the creator; `critic` means only the objective critic "
                        "scored the group. A creator/agent pick is logged "
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

    sub.add_parser("selftest", help="check this install's toolchain end to end")
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
                                 select=args.select, log=not args.no_log,
                                 refine=args.refine, feature_spec=args.feature_spec,
                                 effect_grid=args.effect_grid)
        elif args.cmd == "replay":
            result = cmd_replay(args.dataset, args.limit or None)
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
        elif args.cmd == "selftest":
            result = cmd_selftest()
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
