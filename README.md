# EditApart — AI taste-driven editor (agent preset)

**EditApart** is an automation-faithful editor for the **EditTogether** community.
It takes raw footage, builds a shot inventory, proposes an **EDIT DECISION SCHEMA**
against a declared editorial rubric, renders it deterministically with
ffmpeg/scenedetect (and ImageMagick for photos), critiques the render against the
rubric, and **revises** until it converges.

It does **not** operate a video-editing GUI. It reasons about editorial judgment.
The plugin registers both video and photo tools plus the taste-model tools (15
total) under one identity; the engines are `bin/edit_apart_core.py` (video),
`bin/photo_core.py` (photo) and `bin/taste_model.py` (the per-creator model).

## The core design

- **Schema is the interlingua.** The same EDIT DECISION SCHEMA is the thing you
  train on, the thing you render from, and the thing you critique. No gap
  between "AI intent" and "the render."
- **General grammar is shared; taste is per-creator.** One shared style-brain
  (film-grammar parameters) + a small per-creator identity latent `z_u` in a
  per-creator GGUF. Only the identity is per-creator — the artifact holds `z_u`
  and a digest of the trunk it belongs to, nothing else.
- **Reward is dense + group-relative (GRPO).** The critic scores a render AND
  yields per-element keep/drop/why deltas; `r = overall + λ·Σ(deltas)`. Group
  relative advantage `A_k = (r_k − mean(r))/std(r)`, then a PPO-clip step.
- **Muon on weights, AdamW on the latent.** Muon orthogonalizes momentum for
  2D weight matrices; **never** the identity latent (Newton-Schulz degenerates
  on 1D). AdamW handles `z_u` and all 1D/bias/embed params.
- **Self-play interleaved with history.** `propose_schema` emits a *group* of
  candidate schemas from a deterministic grid; the group is logged, one candidate
  is selected, and the render/vision/creator outcome is logged back against it.
  Training consumes those groups overnight.
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
  propose_schema + rubric ─► GROUP of candidate EDIT DECISION SCHEMAS
               │                    │         (logged to .editapart/taste_samples.jsonl)
               │                    ▼
               │        taste identity scores the group ─► selected schema
               │
    render_schema ─► deterministic ffmpeg render
               │
    critic_edit ─► score + per-element keep/drop/why deltas
               │                    │         (logged as that candidate's reward;
               │                    │          chosen_by=creator|agent = revealed preference)
    revise_schema ─► revised schema ─► re-render (bounded loop)
               │
   train_identity ─► shared style-brain GGUF + per-creator z_u GGUF
```

> **Status of the taste model.** The identity learning is **implemented and wired
> into the loop**, not a scaffold. `propose_schema` builds and logs a group of
> candidates, selects one with the creator identity when it exists, `critic_edit`
> logs the reward for the candidate that was actually rendered, and
> `train_identity` fits the shared style-brain plus that creator's latent `z_u`
> and writes real GGUF artifacts. The measured claims and their limits are in
> [Taste model](#taste-model-wired-measured) below — including what it does *not*
> do yet (trained selector, **not** a fine-tuned language model; video only).

### Photo editing (image-only)

A parallel single-image loop, mounted via the `edit-apart` plugin:
`photo_inspect` → `photo_propose` (rubric → photo EDIT DECISION SCHEMA) →
`photo_render` (ImageMagick) → `photo_critic` (objective + subjective) →
`photo_revise`. It needs **one fewer modality** than video (`image`, not
`video`), and its renderer is **not a hard dependency** (resolved at runtime;
a web-profile in-browser ImageMagick/ffmpeg may supply it).

The taste model is wired in here too, with its own `photo/v1` layout, its own
group log and its own style-brain (`style_brain_photo_v1.gguf`), so a creator
ends up with one identity per modality. Three things make this modality harder
than video, and all three are measured rather than assumed:

- **No temporal axis to lean on**, so the layout synthesises spatial structure:
  crop geometry plus statistics of the region the crop *keeps*, read from a 4×4
  region grid that `photo_inspect` computes from the same downscale as the global
  stats (the global numbers are unchanged).
- **Candidate scoring costs a render.** The photo reward is defined on the
  rendered image, so a K-candidate group is rendered and scored K times
  (video's objective critic is pure arithmetic). `group=1` with no dataset stays
  the legacy, render-free proposal.
- **A quantised reward ties the group.** Exposure was a threshold, so a 6-candidate
  group scored 6 × −0.05, spread 0.0 — exactly zero group-relative signal. The
  reward is now graded (full credit inside the tolerance, linear falloff outside),
  which took that group to spread 0.47 and left 11/18 training groups informative.

Measured on a synthetic 1600×1200 image, two creators with opposite framing taste
(tightest vs loosest crop) sharing one frozen trunk: **10/10 held-out briefs from
the training family are picked differently, 10/10 in the predicted direction**
(mean crop area 0.155 vs 0.389). On **one unseen crop geometry it transfers too —
10/10 over ten distinct briefs — but with a smaller margin**: mean crop area 0.30
vs 0.45, a gap of 0.15 against 0.233 in-distribution. That is the geometry/content
confound of `docs/paper-findings.md` (Finding 4c) narrowing the effect, not
erasing it. (An earlier "0/10, no transfer" reading came from the objective bug
described under **Corrections**; it was an artifact and is withdrawn.)

A four-arm follow-up (`tests/experiment_photo_transfer.py`: {1 image, 3 images} ×
{fixed, varied crop family}, shared probe ladder, matched steps) found the repair
is partial and along the **content** axis, not the geometry axis: only the
3-image/narrow-family arm transferred to an unseen image *and* geometry (3/3,
ratio 0.400), while both varied-geometry arms gave 0/3. A **no-taste control
identity** showed the ordering metric's noise floor is ±0.36 — larger than most
effects measured — so treat these results as suggestive, not established.
`docs/paper-findings.md` Findings 4c/4d has the tables, the power analysis and the
rules (train on the family you will use; include an out-of-family check and a
no-taste control).

## Taste model (wired, measured)

`bin/taste_model.py` is the per-creator model; `bin/train_identity.py` is its CLI.
The loop in `bin/edit_apart_core.py` drives it.

**Architecture.** A shared trunk over 16 named edit features (pacing, shot
selection, tempo, energy, order) feeds a hidden layer that the creator latent
modulates three ways:

```
u = tanh-free ranking score =
      (a ⊙ (1 + tanh(W_s z + b_s))) @ W2 + b2     # latent gates the hidden activations
    + (z @ W_z + b_z)                              # latent output offset
    + x · (z @ W_l)                                # per-creator linear readout over the features
```

`z_u` is the only per-creator tensor, so a creator artifact is ~1 KB while the
trunk is shared. The map is nonlinear in `z` (tanh gates, multiplicative terms)
and interpolatable (the latent is continuous; a midpoint latent scores between
its endpoints — asserted in the test suite).

**Objective.** For each clip the loop logs a group of candidates with the critic's
dense reward. GRPO turns that into group-relative advantages
`A_k = (r_k − mean r)/std r`, and the update is a PPO-clip surrogate on the
group-softmax selection policy, plus an auxiliary pairwise preference loss over
the pick that was actually rendered and a small entropy bonus. Utilities are
standardised *within* the candidate group before the softmax: this keeps the
logits responsive so the latent can always reorder candidates — with an
unbounded-then-squashed output the utilities saturate under the margin objective
and the latent loses all authority over the ordering (measured: 12/12 neutral
briefs identical, versus 12/12 different after the fix).

**Optimizer split.** Muon (quintic Newton-Schulz, coefficients and nesterov form
from the reference Muon implementation) on 2D weights; AdamW on `z_u` and every
1D/bias parameter. Muon's lr is in **spectral-norm units**, so on these small
matrices it must be ~10× below an Adam lr — at the higher lr the model stops
fitting (measured: training accuracy stalls at ~0.5 versus ~0.85).

**Artifacts.** Real GGUF v3 files (F32 tensors + metadata): a shared
`style_brain_video_v1.gguf` and a tiny per-creator `<creator>.gguf` carrying
`z_u`, the feature layout, and the digest of the trunk it was trained against. A
digest mismatch is refused rather than silently scoring with the wrong trunk.

### How the loop feeds it

```bash
# 1. propose a GROUP (logged automatically); the best candidate by the identity
#    is selected, or by the objective critic reward when no identity exists yet
propose_schema inventory=<inv> rubric=<rub> group=4

# 2. render the selected candidate, then log that group's outcome
critic_edit schema=<schema> inventory=<inv> rubric=<rub> group_id=<id> \
            candidate=<idx> chosen_by=creator

# 3. inspect and train
taste_status
train_identity creator=erkin            # writes .editapart/identity.gguf
train_identity creator=erkin freeze_style=true   # later: only z_u moves

# the photo loop is the same protocol on its own log/identity
photo_propose src=<img> inspect=<insp> rubric=<rub> group=4
photo_critic schema=<schema> rubric=<rub> result=<render> group_id=<id> \
             candidate=<idx> chosen_by=agent
taste_status modality=photo
train_identity modality=photo creator=erkin
```

`chosen_by=creator|agent` matters: the pick is logged as a **revealed
preference**, becomes that group's top reward, and gives the trainer a preference
term. A rubric-derived reward alone cannot identify per-user taste: with no
user-dependent term in the objective, two creators with opposite tastes differ on
only **4/12** neutral briefs (p ≈ 0.21 against the unrelated-picks null), versus
**8/12** (p ≈ 0.0006) once the revealed pick is in the objective (margins +3.6 s
vs +7.0…+9.4 s). The override additionally fits the creator's own corpus far
better than the auxiliary loss (0.83/1.00 vs 0.22/0.26), so both mechanisms are
kept; `tests/experiment_revealed_preference.py` reproduces the whole table.

### Measured results

| Claim | Measurement |
| --- | --- |
| Gradients are correct | analytic vs finite differences: **1.7e-08** (model) / **1.7e-09** (full objective, incl. preference term) |
| Objective recovers a hidden preference | held-out pairwise agreement **0.82–0.84** vs 0.50 chance; argmax **0.55–0.86** vs 0.20 chance (3 seeds, 120 synthetic clips) |
| The identity is interpolatable | midpoint latent brackets the endpoints for ≥70% of candidates and moves the mean monotonically |
| Optimizer split holds | Muon touches 2D only, AdamW 1D only (asserted, with witnesses) |
| Artifacts round-trip | GGUF write→read bit-exact; digest mismatch refused |
| The identity changes the loop | two creators on the same shared trunk: **8/12** neutral briefs differed, 8/12 in the predicted direction, **21.4s vs 12.0s** mean selected duration (re-measured after the objective fix below) |
| The revealed pick is what carries taste | ablation (re-measured): with no user term **4/12** briefs differ (p ≈ 0.21, indistinguishable from unrelated picks) vs **8/12** (p ≈ 0.0006) for every mechanism tried; the reward override also fits the creator's own corpus far better (0.83/1.00 vs 0.22/0.26) and widens the margin (+9.4s vs +7.0s) |
| Photo taste transfers, with a narrower margin out of family | two creators, one shared trunk: **10/10** held-out briefs picked differently (10/10 in direction, crop area **0.155 vs 0.389**); on **one unseen crop geometry 10/10** but with a smaller gap (**0.30 vs 0.45**) — the geometry/content confound narrows it |
| The repair is partial and content-driven | 4-arm transfer experiment (re-measured): only the 3-image/narrow-family arm transferred to an unseen image+geometry (tight/loose ratio **0.400**) while A1/A2/B2 showed no effect (1.000); on a *seen* image at the same geometry three of four arms did separate. The ±0.36 ordering-metric noise floor from the no-taste control is a pre-fix measurement |
| The loop is numpy-optional | legacy `propose group=1` and group logging work with numpy blocked; only the model path errors |

Reproduce with `tests/test_taste_model.py` (35 tests, no media needed) and
`tests/test_loop_e2e.py` (real footage; see **Verification**).

### Boundaries (honest)

- **The trained policy is the selector, not the language model.** Training changes
  which candidate the loop picks among the ones this engine generates; it does
  not fine-tune an LLM, and it cannot invent an edit the candidate grid does not
  contain.
- **Video only.** The design's v0 scope is pacing + shot selection + tempo, which
  is what the feature layout covers. Photo taste (colour/composition) is the next
  phase; the model is generic over feature specs, but `photo_propose` still emits
  a single rubric-derived schema and logs nothing.
- **It needs enough clips.** Held-out accuracy is only meaningful once there are
  several logged clips; with one or two the model fits them and generalises
  nothing. `taste_status` reports how many trainable groups are in the log.
- **The trunk is shared, so per-creator training freezes it** (`freeze_style=true`);
  a normal run trains the trunk too and is meant for building the shared model
  from the union of several creators' logs.

## Install

An EditApart install is one directory in the harness's user-preset root. **The
directory name is the preset id**, `agent.cordis.yml` is the composition the
loader owns, and `preset.yml` carries display metadata only — so clone the
repository under the id you want:

```bash
git clone https://github.com/EditTogether/DSH-EditApart.git ~/.dsh/.agent-presets/ai-video-editor
```

Optionally provision the scene-detector/renderer toolchain (a venv, so nothing
touches your system Python):

```bash
cd ~/.dsh/.agent-presets/ai-video-editor && bash setup.sh
```

Restart DeepSeek Harness (or open a new session) and pick **EditApart (AI taste
editor)** in the mode picker. Update with
`git -C ~/.dsh/.agent-presets/ai-video-editor pull`; uninstall by deleting that
directory.

**Requirements.** The harness you install it into — the composition names only
plugins the harness already ships, so there are no npm dependencies to add — plus
`ffmpeg`/`ffprobe` and `scenedetect` for video, and optionally ImageMagick for
photos. No absolute path is baked in anywhere: toolchains resolve `DSH_*` env →
`<preset>/.dshenv` → PATH, so a machine-local install is a `.dshenv` (gitignored)
and never a repository edit.

**Verified as installed.** Running the harness's own preset discovery against a
plain `git clone` of this repository reports id `ai-video-editor`, name
"EditApart (AI taste editor)", and **no `broken` verdict** — every plugin row in
the composition resolves against the runtime's package base. Both files must stay
at the repository root: a directory whose `agent.cordis.yml` is missing still
occupies its id and shows as broken rather than mounting.

This project is discoverable in the DeepSeek Harness ecosystem under the
[`dsh-plugin`](https://github.com/topics/dsh-plugin) topic.

## How to use it

1. Start a session on this preset (select `ai-video-editor` / EditApart in the
   mode picker). The preset mounts its own `edit-apart` plugin, so the tools
   `inventory`, `features`, `propose_schema`, `render_schema`,
   `review_frames`, `critic_edit`, `revise_schema`, `train_identity`,
   `taste_status`, `identity_init`, `photo_inspect`, `photo_propose`,
   `photo_render`, `photo_critic`, and `photo_revise` appear in that session's
   catalog.
2. The bundled `edit-apart` skill drives the loop.
3. Toolchains are **resolved at runtime, not hard-locked**: `ffmpeg`/`ffprobe`/
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
  `KeyError: 'rad2deg'` import crash). `numpy` is a scenedetect dependency anyway
  and is what the taste model runs on; the legacy loop (`propose group=1`) works
  without it.
- `setup.sh` — creates the venv, installs deps, writes `env.sh`/`.dshenv`.

Override keys: `DSH_EDIT_PY`, `DSH_SCENEDETECT`, `DSH_FFMPEG`, `DSH_FFPROBE`,
`DSH_IMAGEMAGICK`.

Taste-model keys: `DSH_EDITAPART_IDENTITY` (per-creator GGUF),
`DSH_EDITAPART_DATASET` (group log), `DSH_EDITAPART_PHOTO_IDENTITY` /
`DSH_EDITAPART_PHOTO_DATASET` (the photo loop's own pair),
`DSH_EDITAPART_DATA` (directory holding them, default `<workspace>/.editapart`),
`DSH_EDITAPART_STYLE` (shared style-brain path) and `DSH_EDITAPART_GROUP`
(candidates per proposal, default 4). The identity and the log are per-workspace
on purpose, so each project accumulates its own creator history.

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
- The exposure/contrast terms are **graded** (see the photo taste section): full
  credit inside the tolerance exactly as before, linear falloff outside it, so a
  candidate that passes is scored identically to the pre-grading critic.

## Corrections

An external review re-ran the suites and reproduced five defects. All are fixed
here, and the behavioural numbers this README reports were re-measured afterwards
(the earlier figures are withdrawn, not silently kept).

1. **The GRPO ratio was not a ratio** (`bin/taste_model.py`). `train()` stored raw
   standardised logits as the ratio baseline, so `rho = exp(log_softmax(s) - s) =
   1/Z` — a per-group constant. Negative-advantage candidates then always fell
   into the clipped branch and their downward push stopped depending on `|A|`
   (measured: an identical 0.0268 for A = −0.1, −2.0 and −8.0), so the objective
   had silently degenerated to positive-only REINFORCE. The telemetry had the same
   bug: `clip_fraction` was a per-call sum divided by the batch count, reported as
   5.9 (a "fraction" of 590%). Fixed by storing `log_softmax(...)` and separating
   the two denominators; `tests/test_taste_model.py::TestObjectiveFidelity` pins
   `rho == 1`, `clip_fraction ∈ [0, 1]`, and that a −2.0 advantage pushes ~10×
   harder than a −0.1 one. **Consequence:** video per-creator separation
   re-measures 12/12 → **8/12** (21.4s vs 12.0s) and the photo out-of-family
   result 0/10 → **10/10 with a narrower margin** (0.30 vs 0.45).
2. **ImageMagick 6 silently killed the photo loop's crop axis**
   (`bin/photo_core.py`). Resolving `convert` and then running `convert identify …`
   fails on IM6, and `_dims` returned `0x0` **silently**: every percent crop became
   `0x0+0+0` and the entire spatial feature family went to zero while
   inspect/propose/render/critic all still reported success. Now the identify
   binary is resolved separately (`DSH_IDENTIFY` > `magick identify` > sibling
   `identify` > PATH), a dimension failure is loud, `propose` refuses a
   dimension-less inspect or an all-identical candidate group, and both cores ship
   a `selftest` conformance check (`python bin/photo_core.py selftest`).
3. **A video render hard-required audio on every input.** One muted clip failed the
   whole graph with "Stream specifier 'a' matched no streams". A silent input now
   gets `anullsrc`, and when *no* input carries audio the silence is emitted
   without `loudnorm` — single-pass `loudnorm` on digital silence divides by zero
   energy and hands the AAC encoder NaN.
4. **`"reward_obj": null` aborted the whole training log** (`float(None)`).
   Unscored candidates are tolerated and skipped, with counters surfaced by
   `taste_status`.
5. **The video E2E logged the wrong candidate's reward**: it critiqued the
   *selected* schema while labelling the reward with `--candidate <pick>`. Candidate
   schemas are now exposed in the group metadata — which is also what the skill's
   "render a different candidate than the selected one" workflow requires — and the
   test asserts the logged reward is the picked candidate's.

Smaller fixes from the same review: schema values that reach the ffmpeg
filtergraph are validated (`_num`/`_ff_crop`), `revise` no longer `KeyError`s on a
schema without `meta`, photo `temperature` actually renders (it was
`-fill … -colorize 0`, a 0% no-op), one temp dir per propose call instead of one
per candidate, `created_at`/`trained_at` artifact metadata made consistent, and an
empty `style_file` no longer resolves to a directory.

## Verification

```bash
# model + video loop unit suite: 36 tests, no media required (~25s)
~/dsh-edit-venv/bin/python tests/test_taste_model.py -v

# photo taste suite: 23 tests (feature tests need no renderer; the grouped and
# per-creator tests need ImageMagick; ~4 min, dominated by candidate renders)
~/dsh-edit-venv/bin/python tests/test_photo_taste.py -v

# end-to-end on real footage (scenedetect + ffmpeg + a real render)
DSH_EDIT_PY=~/dsh-edit-venv/bin/python \
DSH_SCENEDETECT=~/dsh-edit-venv/bin/scenedetect \
DSH_EDITAPART_E2E_SRC=/path/to/footage.mp4 \
~/dsh-edit-venv/bin/python tests/test_loop_e2e.py -v
```

The unit suite checks Newton-Schulz's bounded band and scale invariance, the
optimizer split (Muon never sees a 1D gradient, AdamW never sees a 2D one),
analytic gradients against finite differences, GGUF round-trips and digest
mismatch refusal, dataset folding (including legacy records and broken lines),
learning on a synthetic hidden preference, latent ablation/interpolation,
`group=1` reproducing the legacy schema exactly, critic-reward logging, and
graceful degradation when numpy is unavailable.

The E2E suite builds an inventory from real footage, drives two creators with
different revealed preferences through the real loop, trains one shared trunk
plus two frozen-trunk identities, and asserts that the same neutral brief gets
materially different edits — then renders the taste-selected schema with real
ffmpeg and critiques/revises it. Measured on
`better-com-ceo-roasted-by.mp4` (41.7s, 23 shots): **8/12** briefs differed,
long-take 21.4s vs short-take 12.0s mean selected duration; the rendered
taste-selected edit was 18.75s, critic overall −0.15, 10 → 7 segments after
revise. (These are the post-correction numbers; see **Corrections** below.)

See `skills/edit-apart/SKILL.md` for the full protocol.
