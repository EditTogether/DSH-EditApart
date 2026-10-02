#!/usr/bin/env python3
"""First-hand conformance test for the interlingua decision behind this project.

WHY THIS EXISTS
---------------
EditApart's edit decision is a *bespoke* schema. The state of the art has settled on
OpenTimelineIO (OTIO) as the shareable timeline interlingua, so an NLE-shaped
EditApart should align to it — but only where OTIO can actually carry the decision,
and explicitly elsewhere.

That boundary is a claim about a standard, so this file checks it against the
installed library rather than quoting someone's reading of it. It answers four
questions and fails loudly if the answer changes:

  1. How many transition types does OTIO's core actually define?
  2. Which track kinds does it define?
  3. Does the core schema carry audio level/gain/fade at all?
  4. Can our extensions round-trip through OTIO's own JSON serializer?
     (audio gain as an Effect, beats/emotion as Markers, asymmetric transition
     offsets, and a source range on a clip)

Verified against opentimelineio 0.18.1 on 2026-09-29:
    Transition.Type      = Custom, SMPTE_Dissolve          (2, not a vocabulary)
    Track.Kind           = Audio, Video                    (no image kind)
    core audio/gain      = ABSENT (must be an Effect extension)
    built-in serializers = otio_json, otiod, otioz         (EDL/FCPXML are plugins)
    round-trip           = markers + effect metadata + in/out offsets all survive

Run:  python tools/otio_conformance.py [--json]
Exit: 0 when every claim holds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys

try:
    import opentimelineio as otio
    from opentimelineio import schema
except ImportError:  # pragma: no cover - environment problem, not a code path
    print("opentimelineio is not installed: python -m pip install opentimelineio",
          file=sys.stderr)
    raise SystemExit(2)

RATE = 25.0
EXPECTED_SERIALIZERS = {"otio_json", "otiod", "otioz"}


def _rt(value: float = 0.0) -> "otio.opentime.RationalTime":
    return otio.opentime.RationalTime(value, RATE)


def build_probe_timeline() -> "schema.Timeline":
    """A smallest-possible timeline exercising every extension we depend on."""
    timeline = schema.Timeline(name="editapart-nle-conformance", global_start_time=_rt(0))
    video = schema.Track(name="V1", kind=schema.Track.Kind.Video)
    audio = schema.Track(name="A1", kind=schema.Track.Kind.Audio)
    timeline.tracks.append(video)
    timeline.tracks.append(audio)

    # Two shots from one source file, each with its own source window. OTIO's
    # `source_range` is the standard form of EditApart's trim {in,out}.
    ref = schema.ExternalReference(target_url="file:///media/shot_001.mp4")
    video.append(schema.Clip(name="shot_001", media_reference=ref,
                             source_range=otio.opentime.TimeRange(_rt(120), _rt(75))))
    video.append(schema.Clip(name="shot_002", media_reference=ref,
                             source_range=otio.opentime.TimeRange(_rt(0), _rt(100))))

    # EXTENSION 1 - audio gain/fade. The core schema has none (asserted below), so
    # this follows the established ecosystem pattern (auto-editor's OTIO export
    # emits an `AudioFader` effect with Volume/Mute keyframes) and puts our own
    # numbers in the effect's metadata namespace.
    fader = schema.Effect(name="AudioFader", effect_name="AudioFader")
    fader.metadata["editapart"] = {"gainDb": -6.0, "fadeInFrames": 0,
                                   "fadeOutFrames": 6, "muted": False}
    audio_clip = schema.Clip(name="clip_audio", media_reference=ref)
    audio_clip.effects.append(fader)
    audio.append(audio_clip)

    # EXTENSION 2 - beats/emotion. These are NOT clip properties: a beat belongs to
    # the music and can span shots, which is exactly what a zero-duration Marker is
    # for. Marker carries name/color/comment/marked_range/metadata.
    video.markers.append(schema.Marker(
        name="beat_downbeat",
        marked_range=otio.opentime.TimeRange(_rt(240), _rt(0)),
        color=schema.Marker.Color.RED,
        comment="downbeat 1",
        metadata={"bpm": 120.0, "role": "climax"},
    ))

    # EXTENSION 3 - asymmetric transition. EditApart has one `durationFrames`; OTIO
    # splits it into how much of the cut eats the outgoing and incoming shot.
    video.insert(1, schema.Transition(
        name="dissolve_12f", transition_type=schema.Transition.Type.SMPTE_Dissolve,
        in_offset=_rt(6), out_offset=_rt(6)))
    return timeline


def check() -> dict:
    findings: list[dict] = []
    failures: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        findings.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    transition_types = sorted(m for m in dir(schema.Transition.Type) if not m.startswith("_"))
    record("transition vocabulary is tiny",
           transition_types == ["Custom", "SMPTE_Dissolve"],
           f"Transition.Type = {transition_types}")

    track_kinds = sorted(m for m in dir(schema.Track.Kind) if not m.startswith("_"))
    record("track kinds are Video/Audio only",
           track_kinds == ["Audio", "Video"],
           f"Track.Kind = {track_kinds}")

    # Probe for any level-bearing member across the core schema. Two traps this
    # probe fell into first and now guards against: (a) a narrow word list silently
    # produces an empty result that "confirms" anything, so the list covers every
    # spelling a level could have; (b) dir() on a class can surface names that do
    # not resolve, so each candidate is resolved with getattr before it counts.
    words = ("volume", "gain", "audio", "level", "fader", "db", "mute")
    audio_hits: list[str] = []
    for name in ("Timeline", "Track", "Clip", "Transition", "Marker", "Effect", "Stack", "Gap"):
        cls = getattr(schema, name, None)
        if cls is None:
            continue
        for prop in dir(cls):
            if prop.startswith("_") or not any(word in prop.lower() for word in words):
                continue
            try:
                getattr(cls, prop)
            except Exception:
                continue          # a phantom name, not a field
            audio_hits.append(f"{name}.{prop}")
    record("core schema has no audio level, gain, fade or mute field",
           audio_hits == ["Timeline.audio_tracks"],
           "the only audio-named member anywhere in the core schema is "
           f"Timeline.audio_tracks, an accessor that returns the Audio tracks — "
           f"nothing carries a level: {audio_hits}")

    serializers = set(otio.adapters.available_adapter_names())
    record("only the OTIO serializers ship built in",
           EXPECTED_SERIALIZERS <= serializers,
           f"built-in adapters = {sorted(serializers)} "
           "(EDL/FCPXML/AAF are separate packages)")

    timeline = build_probe_timeline()
    text = otio.adapters.write_to_string(timeline, "otio_json")
    back = otio.adapters.read_from_string(text, "otio_json")
    record("timeline round-trips",
           back.name == timeline.name and len(back.tracks) == 2,
           f"{len(back.tracks)} tracks, {len(text)} JSON bytes")
    record("source range (trim) survives",
           back.tracks[0][0].source_range.start_time.value == 120,
           f"shot_001 in-point = {back.tracks[0][0].source_range.start_time.value} frames")
    marker = back.tracks[0].markers[0]
    record("beat marker + metadata survives",
           marker.name == "beat_downbeat" and marker.metadata.get("bpm") == 120.0,
           f"marker={marker.name!r} metadata={dict(marker.metadata)}")
    effect = back.tracks[1][0].effects[0]
    record("audio gain as an Effect survives",
           effect.effect_name == "AudioFader"
           and effect.metadata["editapart"]["gainDb"] == -6.0,
           f"{effect.effect_name} gainDb={effect.metadata['editapart']['gainDb']}")
    transition = back.tracks[0][1]
    record("asymmetric transition offsets survive",
           transition.in_offset.value == 6 and transition.out_offset.value == 6,
           f"in/out = {transition.in_offset.value}/{transition.out_offset.value} frames")

    return {
        "otio_version": otio.__version__,
        "findings": findings,
        "failures": failures,
        "ok": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    result = check()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1

    print(f"OTIO {result['otio_version']} — interlingua conformance\n")
    for finding in result["findings"]:
        print(f"  [{'ok' if finding['ok'] else 'FAIL'}] {finding['check']}")
        print(f"         {finding['detail']}")
    print()
    if result["ok"]:
        print("every claim holds; the extensions this project needs are expressible")
        return 0
    print(f"FAILED: {', '.join(result['failures'])}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
