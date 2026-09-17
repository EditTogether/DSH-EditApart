#!/usr/bin/env python3
"""Shared engine core for the AI Taste Video Editor preset plugin.

This module is the single implementation of the render->critique->revise loop
that the preset's plugin (`plugins/video-editor.mjs`) shells out to from its
tool `execute` handlers. It is deliberately import-free (stdlib + subprocess
only) so the plugin never needs to import the harness, and deterministic.

The plugin invokes it as:
    edit_apart_core.py <subcommand> <args...>

Subcommands mirror the tools: inventory | features | propose | render |
review_frames | critic | revise. Each reads JSON on stdin and writes JSON to
stdout, so the plugin can pass args through and surface the result directly.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

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
# propose
# --------------------------------------------------------------------------
def cmd_propose(inventory: dict, rubric: dict) -> dict:
    shots = inventory["shots"]
    min_dur = rubric.get("min_shot_dur", 0.8)
    max_dur = rubric.get("max_shot_dur")
    target = rubric.get("target_duration")
    kept = [s for s in shots
            if (not rubric.get("skip_short", True) or s["duration"] >= min_dur)
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


def cmd_critic(schema: dict, inventory: dict, rubric: dict,
               render_dur: float | None, subjective: float | None) -> dict:
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

    p = sub.add_parser("render")
    p.add_argument("src"); p.add_argument("schema"); p.add_argument("out")

    p = sub.add_parser("review_frames")
    p.add_argument("src"); p.add_argument("schema"); p.add_argument("outdir")
    p.add_argument("--size", type=int, default=640)

    p = sub.add_parser("critic")
    p.add_argument("schema"); p.add_argument("inventory"); p.add_argument("rubric")
    p.add_argument("--render-dur", type=float, default=None)
    p.add_argument("--subjective", type=float, default=None)

    p = sub.add_parser("revise")
    p.add_argument("schema"); p.add_argument("critic")

    args = ap.parse_args()
    try:
        if args.cmd == "inventory":
            result = cmd_inventory(args.src, args.threshold, args.min_scene_len)
        elif args.cmd == "features":
            inv = json.loads(args.inventory)
            result = cmd_features(args.src, inv)
        elif args.cmd == "propose":
            result = cmd_propose(json.loads(args.inventory), json.loads(args.rubric))
        elif args.cmd == "render":
            result = cmd_render(args.src, json.loads(args.schema), args.out)
        elif args.cmd == "review_frames":
            result = cmd_review_frames(args.src, json.loads(args.schema), args.outdir, args.size)
        elif args.cmd == "critic":
            result = cmd_critic(json.loads(args.schema), json.loads(args.inventory),
                                json.loads(args.rubric), args.render_dur, args.subjective)
        elif args.cmd == "revise":
            result = cmd_revise(json.loads(args.schema), json.loads(args.critic))
        else:
            ap.error(f"unknown command {args.cmd}")
    except Exception as exc:  # noqa: BLE001 — surfaced to the plugin as a tool error
        print(json.dumps({"error": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
