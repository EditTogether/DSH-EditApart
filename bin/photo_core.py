#!/usr/bin/env python3
"""Photo (single-image) edit engine for the AI Taste Video Editor preset.

This is the image-only analogue of `edit_apart_core.py`. It is deliberately
import-free (stdlib + subprocess) and deterministic, and it needs ONE fewer
input modality than the video core: photo edits read a single IMAGE, so the
general model only needs the `image` modality — no `video` modality and no
video-in/out registration. ImageMagick (`magick`) is the deterministic renderer.

Subcommands (JSON on stdout, {error} + exit 2 on failure):
    photo_inspect SRC              -> per-image features
    photo_propose SRC INSPECT RUBRIC -> photo EDIT SCHEMA
    photo_render SRC SCHEMA OUT    -> deterministic magick render
    photo_critic SCHEMA RUBRIC RESULT [--subjective X] -> score + per-op deltas
    photo_revise SCHEMA CRITIC     -> drop flagged operations
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

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


def _rgb_stats(src: str) -> tuple[list[float], float]:
    """Mean RGB and mean saturation from a small raw RGB downscale."""
    r = run([MAGICK, src, "-resize", "16x16!", "-depth", "8", "rgb:-"], binary=True)
    data = r.stdout if r.returncode == 0 else b""
    if not data:
        return [0.0, 0.0, 0.0], 0.0
    px = [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data) - 2, 3)]
    n = len(px) or 1
    mean = [sum(p[c] for p in px) / n / 255.0 for c in range(3)]
    sat = sum((max(p) - min(p)) / 255.0 for p in px) / n
    return [round(v, 3) for v in mean], round(sat, 3)


def _percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    return sv[min(len(sv) - 1, int(round(p * (len(sv) - 1))))]


def cmd_inspect(src: str) -> dict:
    _ensure_magick()
    dims = _dims(src)
    lumas = list(_luma_samples(src))
    mean_rgb, sat_mean = _rgb_stats(src)
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
    }


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------
def cmd_propose(src: str, inspect: dict, rubric: dict) -> dict:
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
    return {"meta": {"source": src, "intent": intent, "width": inspect["width"], "height": inspect["height"],
                     "generator": "photo_core.propose"},
            "operations": ops}


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
def cmd_critic(schema: dict, rubric: dict, result_path: str, subjective: float | None) -> dict:
    insp = cmd_inspect(result_path)
    target_luma = rubric.get("target_luma", 0.5)
    exposure_ok = abs(insp["mean_luma"] - target_luma) <= 0.15
    contrast_ok = insp["luma_std"] >= rubric.get("min_contrast", 0.12)
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
               "contrast_ok": contrast_ok}
    overall = 0.5 * (exposure_ok + contrast_ok) / 2.0 + 0.5 * dense
    if subjective is not None:
        overall = 0.6 * overall + 0.4 * subjective
    return {"overall_score": round(overall, 3), "metrics": metrics,
            "reward": round(overall + dense, 3), "elements": elements,
            "revise": [e["op"] + "#" + str(i) for i, e in enumerate(elements) if e["decision"] == "drop"][:8]}


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
# dispatch
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect"); p.add_argument("src")
    p = sub.add_parser("propose"); p.add_argument("src"); p.add_argument("inspect"); p.add_argument("rubric")
    p = sub.add_parser("render"); p.add_argument("src"); p.add_argument("schema"); p.add_argument("out")
    p = sub.add_parser("critic"); p.add_argument("schema"); p.add_argument("rubric"); p.add_argument("result")
    p.add_argument("--subjective", type=float, default=None)
    p = sub.add_parser("revise"); p.add_argument("schema"); p.add_argument("critic")

    args = ap.parse_args()
    try:
        if args.cmd == "inspect":
            result = cmd_inspect(args.src)
        elif args.cmd == "propose":
            result = cmd_propose(args.src, json.loads(args.inspect), json.loads(args.rubric))
        elif args.cmd == "render":
            result = cmd_render(args.src, json.loads(args.schema), args.out)
        elif args.cmd == "critic":
            result = cmd_critic(json.loads(args.schema), json.loads(args.rubric), args.result, args.subjective)
        elif args.cmd == "revise":
            result = cmd_revise(json.loads(args.schema), json.loads(args.critic))
        else:
            ap.error(f"unknown command {args.cmd}")
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
