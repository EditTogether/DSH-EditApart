---
name: edit-apart
description: >-
  Use when editing existing footage by taste: build a shot inventory, propose
  an EDIT DECISION SCHEMA against a declared editorial rubric, render it with
  ffmpeg/scenedetect, critique the render against the rubric, revise, and
  optionally learn the creator's identity latent. Replaces any impulse to
  drive a video-editing GUI.
whenToUse: >-
  Pick this whenever the task is "take this footage and edit it" — especially
  when "edited well" is subjective and tied to a creator's taste, rather than
  a mechanical transform (e.g. a fixed cut or a single filter).
---

# AI Taste-Driven Video Editor (method skill)

You are an **editorial judgment** agent, not a GUI operator. You reason in
editorial terms and encode every decision in the **EDIT DECISION SCHEMA**,
which is simultaneously (a) the thing you learn from, (b) the thing you render
from, and (c) the thing you critique.

The toolchain resolves **env-first, never by a hardcoded absolute path**: the
matching `DSH_*` variable (`DSH_FFMPEG`, `DSH_FFPROBE`, `DSH_SCENEDETECT`,
`DSH_EDIT_PY`), then `<preset>/.dshenv`, then PATH. If a run fails because
`scenedetect` is not on PATH it is installed in a venv, set `DSH_SCENEDETECT`
(or `source ./env.sh`) — do not hardcode one machine's path into the preset.

## The loop (always, in this order, bounded)

1. **Shot inventory** — `scenedetect` to find cut points; `ffprobe` per shot
   for duration, luminance (p50/p95), motion energy, audio RMS/peak, and
   on-beat residual (if a beat grid is declared). Emit a JSON shot list.
2. **Declared rubric** — write/read an explicit editorial spec. It is the
   *constraint + intent* layer. It wins on explicit conflict.
3. **Propose schema** — `propose_schema group=K` builds a *group* of candidates
   from a deterministic grid (pacing/target/length knobs); pick shots, order &
   trim them, set transitions, grade, audio, and a `why` per segment. The schema
   must satisfy hard rubric constraints while the identity latent fills the
   underspecified taste. The group is logged automatically; the selected
   candidate is the identity's argmax when an identity exists, else the best
   objective reward. **If you or the creator prefers a DIFFERENT candidate than
   the selected one, render that one and log it with `chosen_by=creator|agent`.**
4. **Render** — deterministic ffmpeg graph from the schema (see renderer).
5. **Critique** — pass the group id and candidate index so the reward lands on
   the candidate that was actually rendered (`chosen_by=creator|agent` when the
   pick was a preference, not the objective winner). The critic has TWO legs. **Objective** (computed, not
   eyeballed): `critic_edit` scores duration/motion/target against the rubric.
   **Subjective** (read from the actual FRAMES): call `review_frames` to
   extract one frame per segment, then LOOK at them and judge keep/drop/why.
   Only the genuine judgment residue is subjective; everything measurable is
   computed. Combine the two into the reward `r = overall + λ·Σ(deltas)`.
6. **Revise** — apply deltas, re-render. Loop is **bounded** (default 3
   passes); converge or stop.

> **Model capability requirement:** the subjective critic leg reads frames, so
> the session model must declare image input. The `your session model`
> model in this environment requires `inputModalities: [text, image]` under
> `llm-deepseek.models` in `~/.dsh/settings.yaml`; without it, the image gate
> rejects with "model does not declare image input." If you hit that error,
> that is a CONFIG gap (declare the modality + restart), not a model limitation.
> A **video-in** general model (e.g. GLM-5.3) instead declares
> `inputModalities: [text, image, video]`; the video capability is per-provider —
> a provider that does not declare video (e.g. pi-ai) cannot carry it, which is
> expected, not a bug. Prefer a video-in model when the task is to read the raw
> clip; fall back to frames for an image-only general model.

> **Why the subjective leg matters:** a heuristic motion check flags a steady
> talking-head beat as "dead air," but that beat is often the critical set-up for
> a punchline (a caption-bearing shot). Use your eyes, not only the luma-motion
> number — this is the difference between "meets the duration rule" and "this is
> the beat worth keeping."

If the task is only "edit this footage once," you still run 1→6 (steps 3 and 5
still log the group and the reward — that is what makes the next session
trainable). If the task is "train the taste," run the learn step:

7. **Learn** — `taste_status` to see how many groups are logged, then
   `train_identity` (shared trunk + this creator's `z_u`; `freeze_style=true` for
   a per-creator run against an existing trunk). Report the held-out metrics it
   returns; they are the only quality claim. Every later `propose_schema` then
   selects with that identity.

## The EDIT DECISION SCHEMA

The schema is the interlingua. Keep it capable enough to hold a real decision,
but do not model the editor — model the *judgment*. A segment:

```jsonc
{
  "meta": { "source": "...", "fps": 30, "intent": "punchy highlight, no dead air" },
  "structure": [
    {
      "shot": "clip_03",
      "trim": { "in": 12.4, "out": 26.1 },
      "retime": 0.0,                        // 0 = normal; ≠0 = speed ramp factor
      "timeline": { "in": 0.0, "out": 10.9 },
      "transition": { "type": "cut", "dur": 0.0, "params": {} },
      "transform": { "scale": 1.0, "crop": null, "position": null },
      "grade": { "lut": null, "eq": {} },
      "audio": { "level": 1.0, "duck": 0.0, "fade": 0.0 },
      "beat": 300.0,                        // grid anchor this cut lands on (optional)
      "overlay": [],                        // subtitles / captions / titles
      "why": "match-on-action; keep energy; don't re-establish location"
    }
  ],
  "globals": { "color": null, "audioMix": {}, "music": { "bed": null, "syncToBeat": false }, "pace": null }
}
```

Rules:
- The `why` field is load-bearing — it is where editorial *reasoning* survives,
  and it makes the learned taste trainable (imitate the decision, not the
  timestamps).
- Every field in the schema is renderable by the renderer; do not invent a
  field the renderer cannot consume.
- The schema may be partial when the rubric leaves taste open; the identity
  latent (below) supplies the default.

## The rubric

Write it as two parts, both explicit:

- **Constraints** — hard, measurable targets: target duration, max shots,
  min/max shot length, "no shot under 0.8s", palette, "cut on the beat at
  ≥80%".
- **Intent** — subjective direction: "build to a peak by 60%", "calm and
  observant", "respect the speaker's energy".

Every measurable constraint should be computable — you compute those yourself;
only genuinely subjective residue gets the critic's judgment. A rubric that is
only "make it good" is void; write it so you could score a render against it.

## The renderer

Deterministic. From a schema, call `render_schema(src, schema, out)` — it emits
one `ffmpeg` invocation that: trims each segment (`trim`/`setpts`), applies
`scale`/`crop`/`fade`/`colorbalance`/`eq`, normalizes audio (`loudnorm`), and
concatenates with the declared transitions via `filter_complex` + `concat`. The
engine (`bin/edit_apart_core.py`) is the reference implementation; the tool is its
facade. Always probe the source first (`ffprobe`) for width/height/fps.

> **Model capability requirement:** the subjective critic leg reads frames, so
> the session model must declare image input. The `your session model`
> model in this environment requires `inputModalities: [text, image]` under
> `llm-deepseek.models` in `~/.dsh/settings.yaml`; without it, the image gate
> rejects with "model does not declare image input." If you hit that error, that
> is a CONFIG gap (declare the modality + restart), not a model limitation.

## Models: video in / video out

The editor has three distinct model portions; **only one of them is the session
model (the "general" portion)**.

- **General portion** — the session model. It sees the footage and produces the
  editorial judgments (inventory reading, schema proposal, critique). This is
  what a *video in/out* model replaces. A video-in model (e.g. **GLM-5.3**, from
  **Z.ai / zai**, an OpenAI-compatible `chat/completions` endpoint at
  `/api/paas/v4`) accepts the **raw clip** natively via the content part
  `{"type":"video_url","video_url":{"url":…}}` (URL or data-URI; one media kind
  per message) instead of pre-extracted frames. When the general model is a
  video-in model, prefer handing it the clip; otherwise keep the
  `review_frames` decomposition to still images for an image-only model.
- **FilmGPT portion** — the shared film-grammar style-brain. Unchanged by the
  model swap.
- **Creator-specific taste model** — the per-creator identity latent `z_u`.
  Unchanged by the model swap.

> **Deployment constraint:** GLM-5.3 is only wise in a **private cloud or local**
> endpoint, not a public-cloud deployment. Use a self-hosted / private
> OpenAI-compatible Z.ai endpoint, not the public API.
>
> **Scope:** GLM-5.3 replaces the **general portion only**. It does not replace
> the FilmGPT portion nor the creator-specific taste model.
>
> **Generation gating:** model-authored video is gated by **plugin/tool
> architecture, not by prompt** — the preset itself never claims to generate
> video (ffmpeg is the deterministic renderer). Fully AI-generated clips come
> from a **separate store plugin** (or the harness `video_generate` tool, e.g.
> grok-imagine-video), which is opt-in by installation. You treat model-authored
> video as in-scope only when that tool/plugin is actually present in the
> session's catalog. When it is, you may generate a clip and **insert it into a
> regular edit** as a `generated` segment (see below).

### AI-generated segment (insertion)

A regular edit's schema `structure` is normally trimmed from the source footage.
You may also **insert** an AI-generated clip as a segment whose source is the
generated asset instead of the source:

```jsonc
{ "generated": true, "asset": "/path/generated_broll.mp4",
  "trim": { "in": 0, "out": 4 }, "timeline": { "in": 12.0, "out": 16.0 },
  "transition": { "type": "cut", "dur": 0, "params": {} },
  "transform": { "scale": 1.0, "crop": null, "position": null },
  "grade": { "lut": null, "eq": {} }, "audio": { "level": 1.0, "duck": 0.0, "fade": 0.0 },
  "overlay": [], "why": "AI-generated transition b-roll after the beat" }
```

Rules: the `asset` is the generated clip path (from `video_generate`); `trim`
selects a range within it; the renderer treats every `generated` segment as its
own input and concatenates it with the source footage — so a generated clip is a
first-class member of the edit, not an overlay. Generated clips should carry
audio (so the audio concat path can place them); a `why` is still required. Only
insert a generated segment when the generation tool/plugin is actually available.

## Photo editing (image-only)

The same taste-driven loop applies to a single **photo**, and it needs **one
fewer modality** than the video core: photo editing reads an **image**, so the
general model only needs `image` input (no `video` modality, no video-in/out
registration). It works with the existing image-capable general model.

Run the same schema→render→critique→revise loop with the photo tools (mounted
via the `edit-apart` plugin, mapped 1:1 onto `bin/photo_core.py`):

- `photo_inspect(src)` — dimensions, luminance (mean/std/p50/p95), saturation,
  mean RGB. The image features you propose from.
- `photo_propose(src, inspect, rubric)` — rubric (intent, saturation, exposure,
  crop region, output width, caption) → **photo EDIT DECISION SCHEMA**
  (`operations`: crop / grade / overlay_text / resize / sharpen).
- `photo_render(src, schema, out)` — deterministic **ImageMagick** render. Caption
  `overlay_text.text` is sanitized by the engine (`%`/`@`/`\` are escaped) so it
  cannot trigger an ImageMagick expression or file-read; keep captions as
  ordinary text.
- `photo_critic(schema, rubric, result, subjective?)` — objective exposure /
  contrast / saturation metrics + per-operation keep/drop/why deltas + reward;
  optionally fold a vision-model `subjective` (0..1).
- `photo_revise(schema, critic)` — drop flagged operations, keep the rest.

A photo schema is an ordered `operations` array; each op carries its `why` (the
load-bearing editorial reason). Every measurable constraint is computed;
genuinely subjective residue (does the composition land?) is judged on the
**result image** by your own eyes — the subjective leg is normative, the
objective metrics are only a hint. Photo editing stays deterministic: the render
is ImageMagick, never an AI paint-over.

**Photo taste (`photo/v1`).** The same identity model is wired into the photo
loop with its own log, its own style-brain (`style_brain_photo_v1.gguf`) and its
own spatial feature layout. Three consequences you must respect:

- **Scoring a candidate costs a render.** The photo reward is defined on the
  rendered image, so `photo_propose group=K` renders and scores K candidates.
  Keep K modest (3–4) unless you need the spread; `group=1` with no dataset is
  the legacy render-free proposal.
- **The objective reward is coarse.** Exposure is a tolerance, so near-duplicate
  candidates tie and a tied group teaches nothing. The reward is graded (full
  credit inside the tolerance, falloff outside), but the grid is what decides
  whether a group carries signal — check `taste_status` for
  `zero_spread_groups` before blaming a training run.
- **A photo preference is spatial, and the SELECTION does not transfer across
  crop geometry even when the ORDERING does.** Crop geometry decides which pixels
  survive, so the crop-area feature is confounded with the content statistics it
  exposes: two creators trained on one family separate 10/10 on held-out briefs
  from that family, and 0/10 on one unseen geometry (ten distinct briefs), while
  their crop-area rank correlation stays strongly signed throughout — the other
  features dominate the argmax. So say which of the two you mean when you claim
  transfer. A four-arm follow-up found the repair is partial and along the
  **content** axis: three images with a narrow family transferred to an unseen
  image and geometry (3/3), both varied-geometry arms did not (0/3), and a
  no-taste control identity showed the ordering metric's noise floor is ±0.36 —
  larger than most of those effects. Practical upshot: train on the image/brief
  family you will use, vary **content** across training groups, and treat an
  out-of-family check plus a no-taste control as part of accepting a photo
  identity (`docs/paper-findings.md`, Findings 4c/4d). **LOOK at the candidate
  renders** before choosing, and log your choice with `chosen_by=agent` when you
  override the selected candidate.

> **Renderer is NOT a hard dependency.** `bin/photo_core.py` resolves ImageMagick
> at runtime (`DSH_IMAGEMAGICK` env, then PATH), and a web-profile **in-browser**
> ImageMagick/ffmpeg (a store plugin) may supply it instead of a local binary.
> Prefer the photo tools; only fall back to a direct command when you know which
> renderer is present.

## The identity / taste

There is **one shared style-brain** (the general film-grammar parameters) and
**one small per-creator identity** (the latent `z_u` inside a per-creator GGUF).
The style-brain is shared; only the identity is per-creator. This is implemented
and wired: `bin/taste_model.py` is the model, `train_identity` the trainer, and
`propose_schema`/`critic_edit` feed it. The trained policy is the **candidate
selector** — it decides which of the proposed edits you would prefer; it is not a
fine-tuned language model.

- **General grammar = base** — film-grammar craft rules. Shared, and the
  general LLM (this session) already carries the craft.
- **Taste = small identity adapter** — the per-creator latent that modulates
  the edit. Learned with **GRPO**, not trained as a large model.
- **Optimizer split** — Muon on the shared 2D weight matrices; **AdamW** on the
  identity latent `z_u` and all 1D/bias/embed params. (Muon's Newton-Schulz
  orthogonalization degenerates on 1D, so never let it touch `z_u`.)
- **Reward** — the critic's score, plus **dense reward shaping** from the
  critic's per-element keep/drop/why deltas: `r = overall_score + λ·Σ(promoted_element_deltas)`.
- **Training** — **group-relative advantage**: sample a *group* of candidate
  schemas per clip, `A_k = (r_k − mean(r))/std(r)`, then a PPO-clip surrogate
  step (this is what `propose_schema group=K` + `train_identity` do for real).
  Interleave **self-play** (the deterministic candidate grid) with **history**
  (the creator's logged prior edits).
- **Revealed preference must be in the objective.** When the creator or the agent
  renders a candidate that is *not* the one the objective reward prefers, log
  `critic_edit … chosen_by=creator|agent`. That pick becomes the group's top
  reward and a preference term in the loss. A rubric-derived reward alone cannot
  identify per-user taste: with no user-dependent term, two creators with
  opposite tastes separate on only **5/12** neutral briefs (direction at chance),
  versus **12/12 in the predicted direction** with it. The preference loss at the
  default weight suffices; the reward override alone is weaker (10/12), so keep
  both.
- **Muon's lr is in spectral-norm units** — on this small trunk it must be ~10×
  below an Adam lr (`0.005`), otherwise it oversteps and stops fitting
  (training accuracy stalls near 0.5 instead of ~0.85).
- **Utilities are standardised per candidate group** before the softmax. That is
  what keeps the latent able to reorder candidates; an unbounded-then-squashed
  output saturates under the margin objective and the identity loses its effect.
- Overnight is fine — renders and local inference are expensive, so do long
  batches and think of the editor as its own data generator.

## Invocation — the bundled tools

This preset ships its own tools (via the `edit-apart` plugin). Use them, not
ad-hoc bash, for the whole loop. They map 1:1 onto the engines (`bin/edit_apart_core.py` for video, `bin/photo_core.py` for photo):

- `inventory(src, threshold?, min_scene_len?)` — scenedetect cut detection →
  shot list. Raise `threshold` for fewer cuts, lower for more.
- `features(src, inventory)` — per-shot luminance (p50/p95), motion, audio
  RMS/peak.
- `propose_schema(inventory, rubric, group?, identity?, select?, no_log?)` —
  rubric → a **GROUP** of candidate EDIT DECISION SCHEMAS (deterministic grid,
  `group=1` = the single legacy schema), logged to the creator's group log, with
  one candidate selected: by the per-creator identity when one exists, else by
  the objective critic reward.
- `render_schema(src, schema, out)` — deterministic ffmpeg render.
- `review_frames(src, schema, outdir, size?)` — one mid-segment frame per
  segment, the input for the subjective (vision) critic leg.
- `critic_edit(schema, inventory, rubric, subjective?, group_id?, candidate?,
  chosen_by?)` — objective metrics + dense per-element keep/drop/why deltas +
  reward; optionally fold a vision-model `subjective` score (0..1) in. With
  `group_id`, the outcome is logged as that candidate's reward — pass
  `chosen_by=creator|agent` when the pick was a human/agent preference rather
  than the objective winner (that is what teaches the identity).
- `revise_schema(schema, critic)` — apply critic verdict, drop + retime.
- `taste_status(identity?, dataset?)` — is there an identity? how many groups are
  logged and trainable?
- `train_identity(creator?, epochs?, freeze_style?, seed?)` — fit the shared
  style-brain + this creator's latent `z_u` (Muon on 2D, AdamW on `z_u`/1D, GRPO
  + preference loss) and write the per-creator GGUF. Reports held-out metrics:
  pairwise agreement (chance 0.5) and argmax accuracy (chance 1/K). Use
  `freeze_style=true` for a per-creator run against the frozen shared trunk.
- `identity_init(creator?)` — zero-latent identity so a new creator can start.

After `identity_init`/`train_identity`, later `propose_schema` calls select with
that identity automatically (override with `select=objective` or `identity=…`).
Treat the reported held-out accuracy as the only quality claim: training more
clips is what makes it meaningful, and `chosen_by=creator` picks are what make it
*yours*.

All take/return JSON strings for the inventory/rubric/schema/critic payloads.
The persona and tool catalog already describe them; prefer these over running
`python bin/edit_apart_core.py ...` directly.

## Failure handling

- If scenedetect over- or under-cuts, re-run with `--threshold` (raise it for
  fewer cuts, lower for more) — do not hand-patch the inventory.
- If a render command fails, read the ffmpeg error and fix the **graph**, not
  the schema, unless the schema is genuinely invalid.
- Keep the loop bounded; report convergence or the final pass's score honestly.

## Do NOT

- Do not open a GUI editor or act on a timeline.
- Do not invent schema fields the renderer can't consume.
- Do not treat "AI taste" as a vibe to improvise — write the rubric and score
  against it.
- Do not use Muon on the identity latent (or on any 1D parameter).
- Do not drop `chosen_by=creator|agent` when a human/agent pick overrode the
  objective winner — that flag is the taste signal.
- Do not call `train_identity` and then present the artifact as "trained taste"
  without the held-out numbers; report the accuracy or say it is untrained.
- Do not reuse a video identity for photos (or vice versa): the layouts differ
  (`video/v1` vs `photo/v1`) and each identity is bound to its own style-brain.
  Use `train_identity modality=photo` / `taste_status modality=photo`.
- Do not assume a photo preference generalises to a new crop geometry or a new
  image family — measure it (`docs/paper-findings.md`, Finding 4).
- Do not present a single-pass guess as a final edit; always critique and
  revise at least once.
- **Do not rely on the objective heuristic alone.** A steady talking head with a
  caption can be the critical set-up for a punchline (dead-air check flags it,
  vision does not). Always `review_frames` (call the tool, then LOOK at the
  frames) before dropping a shot — the vision leg is the normative judgment;
  motion-luma is only a hint, and can be confidently wrong on the highest-value
  beats.
