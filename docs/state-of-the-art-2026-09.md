# State of the art — agent-native video editing, September 2026

Scope: what exists today that is an "NLE analogue" of this repository, what that
field has already commoditised, and where the real gap is — the research behind
[`expansion-plan.md`](expansion-plan.md).

**Method, and its limits.** Project claims come from reading their READMEs and
manifests first-hand; paper claims come from the abstracts (and, for Crayotter's
system paper, the abstract plus the repository it points at); the interlingua claims
were verified by executing code against the installed library
(`../tools/otio_conformance.py`). Star counts are a point-in-time snapshot, not a
quality signal. Anything I did not verify is marked **unverified** at the end rather
than smoothed over.

---

## 1. Three layers, not one

### Layer 1 — agent-drivable NLEs (timeline UI + agent + export)

| Project | What it is | The part worth copying |
| --- | --- | --- |
| [OpenChatCut](https://github.com/0xsline/OpenChatCut) (~2.1k★, AGPL) | Electron/React multitrack NLE; built-in agent + Streamable-HTTP MCP for Codex/Claude Code; Remotion/WebGL preview; local-first | **Draft sessions with an approval gate**: `begin_edit_session` → edits land in an *isolated draft* → `review_edit_session` → applied **atomically as one undo step**, in `manual` or `auto` mode; only draft-safe tools are exposed in a session, because a rejected proposal could not roll back generation/export |
| [Timeline Studio](https://github.com/MartinDelophy/ai-video-editor) (~0.9k★, MIT) | Browser NLE (CapCut-style) with the portable `.timeline` project as source of truth; headless command runner; 21 WebMCP tools | **Versioned command registry**: validate-only runs, dry runs, **idempotent operation IDs**, transactional application, **predicted diffs**, and an agent skill whose `validate_edit_plan.mjs` gates plans before execution |
| [FableCut](https://github.com/ronak-create/FableCut) (~0.7k★) | Zero-dependency browser editor driven by agents; JSON timeline; MCP + REST; live-reloading UI | Lowest-friction agent loop: the editor *is* the API |
| [frontstage](https://github.com/x777/frontstage), [reelo](https://github.com/soysebas-reyes/reelo), [lumen](https://github.com/Mas-inx/lumen-ai-video-editor), [openscene](https://github.com/Theorvane/openscene), [deepvideo](https://github.com/mecrimino/deepvideo), [chai-studio](https://github.com/stardustv01/chai-studio), [kino-seedance-studio](https://github.com/MrFadiAi/kino-seedance-studio) | The long tail: local-first, MCP-driven, Remotion/HyperFrames or FFmpeg render, "you and your agent edit the same timeline" | Not individually instructive; collectively they show this is now a **category**, not a novelty |

### Layer 2 — headless agent-native engines (no UI at all)

| Project | What it is | The part worth copying |
| --- | --- | --- |
| [cutible](https://github.com/plokdalberb-byte/cutible) (MIT) | "Agent-native montage engine": Timeline-as-Data (pydantic, content hash), 14 low + 8 high-level *verbs*, deterministic FFmpeg render + Remotion contour, QC gate (deterministic + VLM), semantic media index (scenes/transcript/VLM/embeddings), multi-agent swarm, OTIO bridge, render farm, MCP (35 tools) | The closest thing to EditApart's philosophy in the wild; its design principles are the field's consensus: **state is data, not pixels**; **verbs return diffs**; **try / inspect / revert** (checkpoint, undo, branch); **deterministic render**; **closed perception loop** |

### Layer 3 — learned editing preference (the layer I expected to be empty, and was wrong about)

| Work | Mechanism | Why it matters here |
| --- | --- | --- |
| **Crayotter: GRPB** — *Learning Long-Horizon Video Editing Agents via Group-Relative Preference Backpropagation*, [arXiv:2608.02694](https://arxiv.org/abs/2608.02694) (2026-08) | Fixing request + materials + constraints turns a subjective objective into an **ordinal comparison among directly comparable alternatives**; same-task rankings become **zero-sum advantages**, redistributed as **bounded credit over semantic editing segments**; a **lagged allocator and guarded transmission** stop a judgment from shaping its own rollout group. 9B model reported above several proprietary systems on AgenticVBench + blinded human eval | This is EditApart's Finding 1 stated at scale — *a global scalar is the wrong objective; group-relative comparison among same-task alternatives is the fix* — and it adds two things we lack: **credit localised to semantic segments** (not one advantage per clip) and an explicit **lag guard** so a group's own judgment never trains on itself |
| **VlogReward** — *Learning Multi-Dimensional Evaluation for Vlog Editing*, [ICML 2026](https://icml.cc/virtual/2026/poster/61914) | A learned multi-dimensional evaluator for vlog edits | Confirms evaluation is the bottleneck, and that "one scalar reward" is the thing to replace |
| [ReelBrain](https://github.com/Q00/ReelBrain) (source-available, local-first Tauri app) | Four editorial personas behind a Showrunner; a **creator Taste Profile** built from explicit Like / Dislike / Skip with provenance; versioned non-destructive drafts; human approval before anything is built or deployed | Two product-level rules worth adopting wholesale: **"memory is a behavioural prior, never evidence"**, and **current steering always overrides stored taste** — plus taste records that are inspectable, editable, disableable, deletable |
| **Crayotter v1** — *Traceable Multi-Agent Workflows for Long-Form Video Editing*, [arXiv:2606.07636](https://arxiv.org/abs/2606.07636) (2026-05) | Retrieval reports, analyses, editing blueprints, scheduler events, tool calls and intermediate renders are **first-class artifacts**, not transient state; resumption, failure diagnosis, artifact preview; best human overall score (3.40/5) among compared systems | Traceability as an architecture, not a log: the thing that makes a long edit debuggable and reviewable |

> **Correction to my own first pass.** I began this survey expecting EditApart's
> learning half to be unoccupied. It is not: GRPB is the research version of the same
> insight, and ReelBrain is the product version of taste memory. The defensible claims
> below are correspondingly narrower.

### Adjacent design work worth reading

[Nomi's `edit-decision-table-for-agent`](https://github.com/aqm857886159/Nomi/blob/main/docs/research/2026-09-07-edit-decision-table-for-agent.md)
is the best engineering treatment I found of "should the agent hand the human a
*table* of edit decisions?". Two of its findings drive this design:

- Surveying CapCut, Descript, Runway, Premiere, Opus Clip, Captions and Fotor, **no
  product gives a reviewable, itemisable, whole-plan decision table** before
  execution. The industry pattern is *AI does it, it lands on the timeline, you fix it
  afterwards*. The two counter-examples are instructive: Descript's filler-word panel
  (a mature itemised review list — for **one** decision type) and Premiere's AI
  Assistant, which chose **stepwise** interception over a whole-plan table.
- Its architectural rule: the table must be a **projection of the operation list**,
  never a second executable schema, and it must be derived from the kernel's
  *executed result*, not patched together from operation JSON — the documented failure
  mode is a preview overlay that silently draws at frame 0 and "tells the user
  something untrue".

---

## 2. Verified conformance of the interlingua (first-hand)

Run `../tools/otio_conformance.py` (opentimelineio 0.18.1) to reproduce:

| Question | Verified answer | Consequence for us |
| --- | --- | --- |
| Transition vocabulary | `Transition.Type` = **Custom, SMPTE_Dissolve** — two constants | A five-word transition vocabulary is an extension, mapped to `Custom` + metadata on export |
| Track kinds | `Track.Kind` = **Audio, Video** | An image/static-shot track is an extension to declare, not a standard track |
| Audio level / gain / fade | **Absent from the core schema** — the only audio-named member anywhere is `Timeline.audio_tracks`, an accessor that returns the Audio tracks; nothing carries a level | Gain/fade/mute must ride an `Effect`; the ecosystem precedent is auto-editor's `AudioFader` with `Volume`/`Mute` |
| Built-in serializers | `otio_json`, `otiod`, `otioz` only | EDL / FCPXML / AAF are separate packages — EDL is an *export* target, and it cannot carry captions, gain or beat marks at all |
| Round-trip of our four needs | trim as `source_range` ✅; gain as `AudioFader` effect metadata ✅; beats as zero-duration `Marker` with `metadata` ✅; asymmetric `in_offset`/`out_offset` ✅ | The design is expressible today; no fork of the standard is required |
| Markers | `name`, `marked_range`, `color`, `comment`, `metadata` | **Beats and emotion already have a standard home** — putting `beats[]` on a clip would be reinventing it wrongly |

---

## 3. Gap analysis

### Already commoditised — do not build these as differentiators
Timeline-as-data; a verb/tool API that returns diffs; deterministic render; MCP/CLI/
headless operation; checkpoint/undo/branch; dry-run + predicted diff + idempotent
operation IDs; OTIO/FCPXML export; semantic media indexing; a VLM/deterministic QC
loop; multi-agent role split; artifact-first traceability. All of it ships today, at
varying polish, in the projects above.

### Genuinely ours (as of this survey)
1. **A per-creator *latent*, not a per-creator prompt or profile.** EditApart learns a
   ~1 KB `z_u` against a shared trunk and stores a **digest of the trunk it was trained
   against**, refusing to score with a mismatched trunk. ReelBrain stores prose taste
   records; GRPB trains a 9B model globally. Neither gives a small, composable,
   trunk-bound per-creator parameter set.
2. **The decision object is the render input and the training datum.** One schema, three
   uses (render, critique, train) — so what is learned is exactly what is executed.
3. **Deterministic candidate groups as the comparison set.** The alternatives the model
   compares are *generated by the system* and logged, so the comparison is auditable
   and reproducible — versus GRPB's ranking of sampled trajectories.
4. **An objective, reproducible critic with a graded reward** (full credit inside the
   tolerance, linear falloff outside) rather than a second model's opinion — the reward
   can be recomputed by anyone from the schema and the inventory.
5. **The trained policy is a *selector*, not a fine-tuned LLM.** No 9B training run is
   required to take effect.
6. **Measurement discipline**: pre-registered probes, a no-taste control identity, an
   unrelated-picks null, and published corrections when a defect invalidated a number.

### Missing — the actual work this project does
1. **An OTIO-shaped interlingua** (ours is bespoke; §2 shows it is a small, verified
   change).
2. **Draft/approval sessions with atomic undo** and only draft-safe tools exposed.
3. **Versioned operation registry with dry-run, idempotency and predicted diffs.**
4. **The reviewable decision table** — the genuinely open product surface, as a
   *projection* of the operation list.
5. **Beats/emotion as markers**, and multi-track reality (music, captions, overlays)
   rather than one flat shot list.
6. **Artifact-first traceability** so a long edit is debuggable.
7. **Credit localised to semantic segments** (per-shot/per-beat), and a lagged
   allocator, per GRPB.

---

## 4. Not verified / open

- Crayotter's **method details beyond the abstract** (credit redistribution maths,
  lagged allocator, the evaluation suite) — the abstract is clear about the shape, not
  the mechanics; the code was not read.
- **VlogReward**: only the ICML poster page, no paper text.
- [US20260148754A1](https://eureka.patsnap.com/patent/US20260148754A1) ("Intelligent
  video editor for creating non-linear editing timeline") — patent text not read.
- Star counts, stars→quality inferences, and the smaller projects' internals.
- Whether any Layer-1 project's "agent edits the timeline" loop **learns across
  sessions** (their READMEs describe prompt/context-driven editing; ReelBrain is the
  only one that documents persistent taste).
