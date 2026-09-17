# EditApart — AI taste-driven editor (agent preset)

**EditApart** is an automation-faithful editor for the **EditTogether** community.
It takes raw footage, builds a shot inventory, proposes an **EDIT DECISION SCHEMA**
against a declared editorial rubric, renders it deterministically with
ffmpeg/scenedetect (and ImageMagick for photos), critiques the render against the
rubric, and **revises** until it converges.

It does **not** operate a video-editing GUI. It reasons about editorial judgment.
The plugin registers both video and photo tools (12 total) under one identity;
the engines are `bin/edit_apart_core.py` (video) and `bin/photo_core.py` (photo).

## The core design

- **Schema is the interlingua.** The same EDIT DECISION SCHEMA is the thing you
  train on, the thing you render from, and the thing you critique. No gap
  between "AI intent" and "the render."
- **General grammar is shared; taste is per-creator.** One shared style-brain
  (film-grammar parameters) + a small per-creator identity latent `z_u` in a
  per-creator GGUF. Only the identity is per-creator.
- **Reward is dense + group-relative (GRPO).** The critic scores a render AND
  yields per-element keep/drop/why deltas; `r = overall + λ·Σ(deltas)`. Group
  relative advantage `A_k = (r_k − mean(r))/std(r)`, then a PPO-clip step.
- **Muon on weights, AdamW on the latent.** Muon orthogonalizes momentum for
  2D weight matrices; **never** the identity latent (Newton-Schulz degenerates
  on 1D). AdamW handles `z_u` and all 1D/bias/embed params.
- **Self-play interleaved with history.** Sample candidate schemas from the
  current policy + reuse the creator's logged prior edits, overnight.
- **Three model portions, and only one is the session model.** General (the
  session model: reads footage, proposes, critiques) · FilmGPT style-brain
  (shared film grammar) · creator taste (`z_u`). A video-in/out model like
  **GLM-5.3** (Z.ai / zai, OpenAI-compatible) replaces the **general portion
  only** — it reads the raw clip via `{"type":"video_url","video_url":{"url":…}}`
  instead of pre-extracted frames; the FilmGPT and taste portions are untouched.
- **Model-authored video is opt-in architecture, not a preset claim.** ffmpeg
  renders the final cut (`render_schema`). Fully AI-generated clips come from a
  **separate store plugin** (or the harness `video_generate` tool, e.g.
  grok-imagine-video), installed opt-in. When present, you may generate a clip
  and **insert it into a regular edit** as a `generated` segment (source + AI
  clip are separate inputs the renderer concatenates). The preset itself never
  claims to generate video.
- **Photo editing is image-only — one fewer modality.** The same taste-driven
  loop edits a single image via the `edit-apart` plugin (`bin/photo_core.py`),
  which needs only `image` input (no `video` modality, no video-in/out
  registration), so it works with the existing image-capable general model. The
  ImageMagick renderer is resolved at runtime (`DSH_IMAGEMAGICK`/PATH) and is
  **not a hard dependency** — a web-profile in-browser renderer may supply it.

## The pipeline

The editor's tools (mounted as a preset plugin at
`plugins/edit-apart.mjs`) are the thin facade; `bin/edit_apart_core.py` (video)
and `bin/photo_core.py` (photo) are the deterministic engines they shell out to.

```
source ─► inventory ─► shot inventory
               │
     features ─► per-shot luminance/motion/audio
               │
   propose_schema + rubric ─► EDIT DECISION SCHEMA
               │
    render_schema ─► deterministic ffmpeg render
               │
    critic_edit ─► score + per-element keep/drop/why deltas
               │
   revise_schema ─► revised schema ─► re-render (bounded loop)
```

> **Status of the taste model.** The 12 tools above are implemented and the
> render→critique→revise loop is what the plugin actually runs. The **identity
> learning** (`bin/train_identity.py`, GRPO + Muon/AdamW) is a **runnable
> research scaffold** — it implements the objective and the optimizer split
> against a toy model so the step is verifiable, but it is **not wired into the
> tool loop and does not yet produce a real per-creator model.** Treat it as the
> next milestone, not a shipped feature.

### Photo editing (image-only)

A parallel single-image loop, mounted via the `edit-apart` plugin:
`photo_inspect` → `photo_propose` (rubric → photo EDIT DECISION SCHEMA) →
`photo_render` (ImageMagick) → `photo_critic` (objective + subjective) →
`photo_revise`. It needs **one fewer modality** than video (`image`, not
`video`), and its renderer is **not a hard dependency** (resolved at runtime;
a web-profile in-browser ImageMagick/ffmpeg may supply it).

## How to use it

1. Provision the toolchain (see **Portable install** below) or make sure
   `ffmpeg`/`scenedetect` are on PATH.
2. Start a session on this preset (select `ai-video-editor` / EditApart in the
   mode picker). The preset mounts its own `edit-apart` plugin, so the tools
   `inventory`, `features`, `propose_schema`, `render_schema`,
   `review_frames`, `critic_edit`, `revise_schema`, `photo_inspect`,
   `photo_propose`, `photo_render`, `photo_critic`, and `photo_revise` appear in
   that session's catalog.
3. The bundled `edit-apart` skill drives the loop.
4. Toolchains are **resolved at runtime, not hard-locked**: `ffmpeg`/`ffprobe`/
   `scenedetect`/Python (video) and ImageMagick (photo) come from the matching
   `DSH_*` env ( `DSH_FFMPEG`, `DSH_FFPROBE`, `DSH_SCENEDETECT`, `DSH_EDIT_PY`,
   `DSH_IMAGEMAGICK`), then `.dshenv`, then PATH. A web-profile **in-browser**
   ffmpeg / ImageMagick (a store plugin) can supply them instead of local
   binaries.

## Portable install (public / another machine)

The code never hardcodes an absolute path. Toolchains resolve **env `DSH_*` →
`<preset>/.dshenv` → PATH**, so the preset runs on any machine with ffmpeg +
scenedetect. To provision a machine where scenedetect lives in a venv (not on
PATH):

```bash
bash setup.sh          # creates ~/dsh-edit-venv, installs scenedetect, writes env.sh
source ./env.sh        # makes DSH_EDIT_PY / DSH_SCENEDETECT available for this shell
# OR, if the tools are already installed:
#   cp env.sh.example env.sh && source ./env.sh
# setup.sh also writes .dshenv so the plugin auto-loads paths without sourcing.
```

- `env.sh.example` — portable template; copy to `env.sh` (gitignored) and edit.
- `env.sh` — **generated, machine-local, gitignored**; exports the `DSH_*` vars.
- `.dshenv` — machine-local paths the plugin auto-loads when the matching
  `DSH_*` env is not already set (gitignored).
- `requirements.txt` — `scenedetect` (+ `numpy<2` to avoid the opencv
  `KeyError: 'rad2deg'` import crash).
- `setup.sh` — creates the venv, installs deps, writes `env.sh`/`.dshenv`.

Override keys: `DSH_EDIT_PY`, `DSH_SCENEDETECT`, `DSH_FFMPEG`, `DSH_FFPROBE`,
`DSH_IMAGEMAGICK`.

> The plugin is **import-free** on purpose: a user-preset relative-mounted
> module cannot import `@deepseek-ai/*` (its internal imports resolve against
> the preset dir, not the harness). It hand-builds `ToolDefinition`s and calls
> `ctx.tools.register()`, shelling out to `bin/edit_apart_core.py` /
> `bin/photo_core.py`.

## Trust & security

- **No shell command injection.** The engines launch processes with **list-argv**
  `subprocess.run([...])` and the plugins with `spawn(PY, [CORE, ...args])` —
  never a shell string, never `shell=True`/`os.system`. Paths and schema values
  are passed as literal arguments, so a value like `x; rm -rf /` cannot be
  interpreted by a shell. The `;`-joined ffmpeg `filter_complex` string is a
  single argv element, not shell syntax.
- **Caption text is sanitized.** ImageMagick `-annotate`/`-caption`/`-label`
  interpret `%[...]`/`%[fx:...]` format escapes and a leading `@` as "read text
  from file". The photo engine escapes `%`/`@`/`\` to visually-identical
  fullwidth codepoints (`_escape_annotate_text`) so untrusted caption text cannot
  trigger an IM expression or file read — independent of the host ImageMagick
  `policy.xml`. (The system package policy already disables `@` file-reads on
  many images, but the code no longer relies on that.)
- **Paths are trusted tool args.** `src`, `out`, `outdir`, and generated `asset`
  paths are operator/model-supplied and passed straight to ffmpeg/scenedetect.
  Within DSH the sandbox scopes writes; if you run the engines **outside** DSH,
  scope the process so a crafted `out`/`outdir` cannot write outside your
  intended directory.
- **`.dshenv` is a trust boundary.** The plugins load `<preset>/.dshenv` and set
  `process.env` (only when the matching `DSH_*` env isn't already set). Whoever
  can write that file controls `DSH_EDIT_PY`/`DSH_SCENEDETECT`, hence the
  executed interpreter. Keep it out of version control (it is gitignored) and
  treat it as operator-owned.

## Video reference run (verified)

On `better-com-ceo-roasted-by.mp4` (41.7s, 23 shots):
- Inventory: 23 shots.
- Rubric: brisk pace, shots 0.9–4.0s, target 20s.
- Schema: 11 segments (dropped a 0.58s shot). Render: valid 20s 1080p mp4.
- **Objective critic**: overall 0.3, reward −0.1, flagged clip_001/clip_002 as
  "low motion / dead air." **This was wrong** — a motion-luma heuristic called
  the set-up ("Well, when are you going") and the first punchline (Ramsay,
  "and start showing some") dead air, and would have dropped the two best beats.
- **Vision critic (normative)**: overall 0.62. Keep clip_001/clip_002 (the
  set-up + punchline, caption-bearing and high-value); trim/cut clip_005 (a long
  static 3.8s hold); consider reordering the "BETTER.COM EMPLOYEES" cutaway
  (clip_010/011) earlier as an establishing subject card.

The lesson baked into the skill: **the vision leg (frame inspection) is
normative; motion-luma is only a hint and can be confidently wrong on the
highest-value shots.** Always `review_frames` before dropping a shot.

## Photo reference run (verified)

On a synthetic 1800×1200 landscape (`photo.jpg`), rubric = warm punch, saturation
1.15, crop 600×500, width 1200, target_luma 0.52:
- Inspect: mean_luma 0.264, luma_std 0.19, p50 0.18/p95 0.835, saturation 0.132,
  mean_rgb [0.214, 0.277, 0.288].
- Schema: crop → grade → resize.
- Render: 1200×1000. Critic: overall 0.5, exposure_ok/contrast_ok true,
  reward 0.5, all ops kept; revise kept crop/grade/resize.

See `skills/edit-apart/SKILL.md` for the full protocol.
