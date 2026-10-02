#!/usr/bin/env python3
"""Generate the LUTs this engine ships (`luts/*.cube`).

A `.cube` file is text: `TITLE`, `LUT_3D_SIZE n`, then n^3 RGB triples with RED
varying fastest. Keeping them generated (rather than opaque binaries) means the
grade they apply is reviewable in a diff and reproducible from this script:

    python tools/make_luts.py            # write luts/*.cube
    python tools/make_luts.py --check    # fail if the files on disk differ

`tests/test_taste_model.py::TestRenderedEffectContract` calls this with `--check`, so
a hand-edited LUT that does not match its generator is caught.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

SIZE = 8
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "luts"

clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731 - a one-line colour clamp


def warm(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Lift red, ease blue, keep a little contrast."""
    return clamp(0.02 + r * 1.06), clamp(0.01 + g * 1.01), clamp(b * 0.92 - 0.01)


def cool(r: float, g: float, b: float) -> tuple[float, float, float]:
    """The mirror image of `warm`."""
    return clamp(r * 0.92 - 0.01), clamp(0.01 + g * 1.01), clamp(0.02 + b * 1.06)


def render(title: str, fn) -> str:
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {SIZE}", ""]
    for blue in range(SIZE):
        for green in range(SIZE):
            for red in range(SIZE):
                values = fn(red / (SIZE - 1), green / (SIZE - 1), blue / (SIZE - 1))
                lines.append(" ".join(f"{clamp(v):.6f}" for v in values))
    return "\n".join(lines) + "\n"


def outputs() -> dict[pathlib.Path, str]:
    return {OUT / "warm.cube": render("warm", warm),
            OUT / "cool.cube": render("cool", cool)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = outputs()
    if args.check:
        stale = [p for p, text in files.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        for path in stale:
            print(f"stale or missing: {path}", file=sys.stderr)
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
