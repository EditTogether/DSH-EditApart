// EditApart — preset-owning tool plugin for the EditTogether community.
//
// A single import-free plugin mounting ALL of the preset's editing capability:
//   * 7 video tools  (inventory, features, propose_schema, render_schema,
//                     review_frames, critic_edit, revise_schema) -> bin/edit_apart_core.py
//   * 5 photo tools  (photo_inspect, photo_propose, photo_render, photo_critic,
//                     photo_revise)                           -> bin/photo_core.py
//   * 3 taste-model tools (train_identity, taste_status, identity_init)
//                     -> bin/edit_apart_core.py -> bin/taste_model.py
//   * 1 guarded model-routing hook (video-in/out general model) -> opt-in via env.
//
// Mounted by the `ai-video-editor` (EditApart) preset via a RELATIVE path
// (`name: ./plugins/edit-apart.mjs`), so it is import-free on purpose: a
// user-preset relative-mounted module resolves its internal imports against the
// preset dir and cannot `import { defineTool } from '@deepseek-ai/dsh-tools'`.
// It hand-builds plain ToolDefinitions and registers them with
// `ctx.tools.register()`, passing JSON-Schema directly (no defineTool
// normalization). The actual work is done by the import-free engines, invoked
// via child_process. The tool names are generic and shared with any future
// human-in-the-loop mode — only the PRODUCT identity is EditApart.
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { readFileSync, existsSync } from 'node:fs'

export const name = 'edit-apart'
export const inject = ['tools', 'systemPrompt']

const HERE = dirname(fileURLToPath(import.meta.url))

// Load machine-local toolchain paths from .dshenv (next to this preset) ONLY
// when the corresponding env var is not already set, so an operator can either
// export DSH_* or rely on the file. Path resolution stays env-first, then this
// file, then PATH. The code never hardcodes an absolute path.
function loadDSHEnv() {
  const p = join(dirname(HERE), '.dshenv')
  try {
    for (const ln of readFileSync(p, 'utf8').split(/\r?\n/)) {
      const m = /^([A-Z0-9_]+)=(.*)$/.exec(ln.trim())
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '')
    }
  } catch { /* no .dshenv — rely on env / PATH */ }
}
loadDSHEnv()

// Python env-first, then .dshenv, then PATH. On a machine with scenedetect /
// ImageMagick in a venv (not on PATH), .dshenv or DSH_EDIT_PY provides it.
const PY = process.env.DSH_EDIT_PY || 'python3'
const VIDEO_CORE = join(HERE, '..', 'bin', 'edit_apart_core.py')
const PHOTO_CORE = join(HERE, '..', 'bin', 'photo_core.py')

// Where the per-creator taste model lives. The identity (a GGUF holding only the
// creator's latent z_u) and the group log the editor writes are per-WORKSPACE by
// default, so each project accumulates its own creator history. Override with
// DSH_EDITAPART_IDENTITY / DSH_EDITAPART_DATASET / DSH_EDITAPART_DATA.
const DATA_DIR = process.env.DSH_EDITAPART_DATA || join(process.cwd(), '.editapart')
const IDENTITY = process.env.DSH_EDITAPART_IDENTITY || join(DATA_DIR, 'identity.gguf')
const DATASET = process.env.DSH_EDITAPART_DATASET || join(DATA_DIR, 'taste_samples.jsonl')
const STYLE = process.env.DSH_EDITAPART_STYLE || ''
// How many candidate schemas a proposal explores. >1 turns propose_schema into a
// group proposal: the group is logged, the best candidate is selected by the
// creator identity when one exists (otherwise by the objective critic reward).
const GROUP_DEFAULT = String(process.env.DSH_EDITAPART_GROUP || '4')

// The photo loop keeps its own log and identity: its feature layout is spatial
// (photo/v1) and its reward is defined on a RENDERED image, so a candidate group
// there costs one render per candidate. A creator therefore ends up with one
// identity per modality, each bound to its own style-brain.
const PHOTO_IDENTITY = process.env.DSH_EDITAPART_PHOTO_IDENTITY || join(DATA_DIR, 'photo_identity.gguf')
const PHOTO_DATASET = process.env.DSH_EDITAPART_PHOTO_DATASET || join(DATA_DIR, 'photo_samples.jsonl')
const MODALITY = {
  video: { core: () => VIDEO_CORE, identity: IDENTITY, dataset: DATASET, train: 'train' },
  photo: { core: () => PHOTO_CORE, identity: PHOTO_IDENTITY, dataset: PHOTO_DATASET, train: 'taste_train' },
}
const pickModality = (v) => (v === 'photo' ? MODALITY.photo : MODALITY.video)

// Run an engine; it emits JSON on stdout and {error} on failure, exit 2 on a
// caught engine error.
function runCore(core, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(PY, [core, ...args], { stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (d) => { stdout += d })
    child.stderr.on('data', (d) => { stderr += d })
    child.on('error', reject)
    child.on('close', (code) => {
      try {
        const parsed = JSON.parse(stdout || '{}')
        if (code !== 0 || parsed.error) {
          return reject(new Error(parsed.error || stderr || `engine exit ${code}`))
        }
        resolve(parsed)
      } catch (e) {
        reject(new Error(`engine output not JSON: ${stderr || stdout}`))
      }
    })
  })
}

const text = (t) => [{ type: 'text', text: t }]
const jsonText = (obj) => text(JSON.stringify(obj, null, 2))

// The generic JSON-Schema output root, shared by every tool.
const outSchema = { schema: { type: 'object', additionalProperties: true }, render: (a, v) => jsonText(v) }

// --- model routing ---------------------------------------------------------
// ARCHITECTURE-ONLY AS SHIPPED. To opt in, set DSH_AI_VIDEO_MODEL_PROVIDER and
// DSH_AI_VIDEO_MODEL to a REGISTERED provider/model that declares `inputModalities`
// including `video`, then set DSH_AI_VIDEO_MODEL_ROUTE=1 (or remove the guard).
// Until then this hook is inert and the session uses the default model. It hooks
// the agent-scoped `agent/request` waterfall to override the general-portion
// model for THIS agent without changing the global agent-default-model.
const PROVIDER = process.env.DSH_AI_VIDEO_MODEL_PROVIDER ?? ''
const MODEL = process.env.DSH_AI_VIDEO_MODEL ?? ''
const MODEL_ROUTE_ENABLED = process.env.DSH_AI_VIDEO_MODEL_ROUTE === '1'

export function apply(ctx) {
  const tools = [
    // ---- video ----
    {
      name: 'inventory',
      description: 'Detect shot boundaries in a source video with scenedetect (0.7.1) and return a shot list {shot,start,end,duration}.',
      parameters: {
        type: 'object',
        properties: {
          src: { type: 'string', description: 'Source video path.' },
          threshold: { type: 'number', description: 'Cut detection threshold (higher = fewer cuts). Default 27.' },
          min_scene_len: { type: 'number', description: 'Minimum scene length in seconds. Default 0.6.' },
        },
        required: ['src'],
      },
      output: outSchema,
      execute: (args) => runCore(VIDEO_CORE, ['inventory', args.src,
        '--threshold', String(args.threshold ?? 27),
        '--min-scene-len', String(args.min_scene_len ?? 0.6)]),
    },
    {
      name: 'features',
      description: 'Enrich a shot inventory with per-shot luminance (p50/p95), motion, and audio RMS/peak. Takes a source and an inventory JSON string.',
      parameters: {
        type: 'object',
        properties: {
          src: { type: 'string', description: 'Source video path.' },
          inventory: { type: 'string', description: 'Inventory JSON (stringified) to enrich.' },
        },
        required: ['src', 'inventory'],
      },
      output: outSchema,
      execute: (args) => runCore(VIDEO_CORE, ['features', args.src, args.inventory]),
    },
    {
      name: 'propose_schema',
      description: 'Propose an EDIT DECISION SCHEMA from a shot inventory against a rubric (pacing + shot-selection; v0). With group>1 it proposes a GROUP of candidate schemas, logs the group for taste training, and selects the candidate preferred by the per-creator identity when one exists (otherwise the best objective critic reward).',
      parameters: {
        type: 'object',
        properties: {
          inventory: { type: 'string', description: 'Inventory JSON (stringified).' },
          rubric: { type: 'string', description: 'Rubric JSON (stringified): min_shot_dur, max_shot_dur, target_duration, pace.' },
          group: { type: 'number', description: `Candidate schemas to consider (1 = legacy single schema). Default ${GROUP_DEFAULT}.` },
          identity: { type: 'string', description: 'Per-creator taste GGUF to select with. Default: the workspace identity if it exists.' },
          select: { type: 'string', enum: ['auto', 'taste', 'objective'], description: 'auto (taste if an identity exists), taste, or objective.' },
          no_log: { type: 'boolean', description: 'Do not append this group to the training log.' },
        },
        required: ['inventory', 'rubric'],
      },
      output: outSchema,
      execute: (args) => {
        const argv = ['propose', args.inventory, args.rubric,
          '--group', String(args.group ?? GROUP_DEFAULT), '--dataset', DATASET]
        const identity = args.identity ?? IDENTITY
        if (identity && existsSync(identity)) argv.push('--identity', identity)
        if (STYLE) argv.push('--style', STYLE)
        if (args.select) argv.push('--select', String(args.select))
        if (args.no_log) argv.push('--no-log')
        return runCore(VIDEO_CORE, argv)
      },
    },
    {
      name: 'render_schema',
      description: 'Render an EDIT DECISION SCHEMA to a deterministic ffmpeg mp4 (v0: hard cuts, trim, scale, optional eq, loudnorm).',
      parameters: {
        type: 'object',
        properties: {
          src: { type: 'string', description: 'Source video path.' },
          schema: { type: 'string', description: 'EDIT DECISION SCHEMA JSON (stringified).' },
          out: { type: 'string', description: 'Output mp4 path.' },
        },
        required: ['src', 'schema', 'out'],
      },
      output: outSchema,
      execute: (args) => runCore(VIDEO_CORE, ['render', args.src, args.schema, args.out]),
    },
    {
      name: 'review_frames',
      description: 'Extract one mid-segment frame per schema segment — the input for the vision (subjective) critic leg.',
      parameters: {
        type: 'object',
        properties: {
          src: { type: 'string', description: 'Source video path.' },
          schema: { type: 'string', description: 'EDIT DECISION SCHEMA JSON (stringified).' },
          outdir: { type: 'string', description: 'Directory to write frame_<shot>.png.' },
          size: { type: 'number', description: 'Output width in px. Default 640.' },
        },
        required: ['src', 'schema', 'outdir'],
      },
      output: outSchema,
      execute: (args) => runCore(VIDEO_CORE, ['review_frames', args.src, args.schema, args.outdir,
        '--size', String(args.size ?? 640)]),
    },
    {
      name: 'critic_edit',
      description: 'Score a rendered edit + its schema against a rubric: objective metrics (computed) + per-element keep/drop/why deltas + dense reward. Optionally fold in a vision-model subjective score.',
      parameters: {
        type: 'object',
        properties: {
          schema: { type: 'string', description: 'EDIT DECISION SCHEMA JSON (stringified).' },
          inventory: { type: 'string', description: 'Enriched inventory JSON (stringified).' },
          rubric: { type: 'string', description: 'Rubric JSON (stringified).' },
          subjective: { type: 'number', description: '0..1 vision-model judgment to fold in (optional).' },
          group_id: { type: 'string', description: 'Group id from propose_schema: logs this outcome as the group\'s reward.' },
          candidate: { type: 'number', description: 'Which candidate of the group this critique is for (default 0).' },
          chosen_by: { type: 'string', enum: ['critic', 'creator', 'agent'], description: 'Who picked this candidate. A creator/agent pick is logged as a REVEALED preference and becomes the group\'s top reward.' },
        },
        required: ['schema', 'inventory', 'rubric'],
      },
      output: outSchema,
      execute: (args) => {
        const argv = ['critic', args.schema, args.inventory, args.rubric]
        if (args.subjective != null) argv.push('--subjective', String(args.subjective))
        if (args.group_id) {
          argv.push('--dataset', DATASET, '--group-id', String(args.group_id),
                    '--candidate', String(args.candidate ?? 0),
                    '--chosen-by', String(args.chosen_by ?? 'critic'))
        }
        return runCore(VIDEO_CORE, argv)
      },
    },
    {
      name: 'revise_schema',
      description: 'Apply a critic verdict to a schema: drop segments the critic marked for revision and retime the kept sequence.',
      parameters: {
        type: 'object',
        properties: {
          schema: { type: 'string', description: 'EDIT DECISION SCHEMA JSON (stringified).' },
          critic: { type: 'string', description: 'Critic JSON (stringified) with per-element keep/drop decisions.' },
        },
        required: ['schema', 'critic'],
      },
      output: outSchema,
      execute: (args) => runCore(VIDEO_CORE, ['revise', args.schema, args.critic]),
    },
    // ---- taste model ----
    {
      name: 'train_identity',
      description: 'Train the per-creator taste identity from the groups the loop has logged: one shared style-brain trunk (Muon on 2D weights) plus this creator\'s latent z_u (AdamW), optimised by a GRPO group-relative PPO-clip objective with an auxiliary preference loss over revealed picks. Returns held-out ranking metrics and writes the per-creator GGUF.',
      parameters: {
        type: 'object',
        properties: {
          modality: { type: 'string', enum: ['video', 'photo'], description: 'Which loop\'s log/identity to train (default video). Photo uses the photo/v1 layout and its own style-brain.' },
          creator: { type: 'string', description: 'Creator name recorded in the artifact. Default "default".' },
          epochs: { type: 'number', description: 'Training epochs (default 150).' },
          freeze_style: { type: 'boolean', description: 'Train ONLY z_u against the frozen shared trunk (the per-creator step; requires an existing style-brain).' },
          holdout_every: { type: 'number', description: 'Hold out every Nth clip for the reported held-out metrics.' },
          seed: { type: 'number', description: 'Deterministic seed.' },
          verbose: { type: 'boolean', description: 'Per-epoch progress on stderr.' },
        },
      },
      output: outSchema,
      execute: (args) => {
        const m = pickModality(args.modality)
        const argv = [m.train, '--dataset', m.dataset, '--identity', m.identity]
        if (STYLE) argv.push('--style', STYLE)
        if (args.creator) argv.push('--creator', String(args.creator))
        if (args.epochs != null) argv.push('--epochs', String(args.epochs))
        if (args.freeze_style) argv.push('--freeze-style')
        if (args.holdout_every != null) argv.push('--holdout-every', String(args.holdout_every))
        if (args.seed != null) argv.push('--seed', String(args.seed))
        if (args.verbose) argv.push('--verbose')
        return runCore(m.core(), argv)
      },
    },
    {
      name: 'taste_status',
      description: 'Report the state of the wired-in taste loop: whether the per-creator identity exists, how many groups/clips have been logged, how many are trainable, and the reward spread.',
      parameters: { type: 'object', properties: {
        modality: { type: 'string', enum: ['video', 'photo'], description: 'Which loop to report on (default video).' },
        identity: { type: 'string', description: 'Identity GGUF (default: the workspace identity for that modality).' },
        dataset: { type: 'string', description: 'Group log (default: the workspace log for that modality).' },
      } },
      output: outSchema,
      execute: (args) => {
        const m = pickModality(args.modality)
        return runCore(m.core(), ['taste_status', '--identity', args.identity ?? m.identity,
                                  '--dataset', args.dataset ?? m.dataset])
      },
    },
    {
      name: 'identity_init',
      description: 'Create a zero-latent per-creator identity bound to a (possibly new) shared style-brain, so a brand-new creator can run the loop immediately and train from zero.',
      parameters: { type: 'object', properties: {
        modality: { type: 'string', enum: ['video', 'photo'], description: 'Which modality\'s identity to create (default video; photo defaults to the photo/v1 layout).' },
        creator: { type: 'string', description: 'Creator name. Default "default".' },
        identity: { type: 'string', description: 'Identity GGUF to create (default: the workspace identity for that modality).' },
      } },
      output: outSchema,
      execute: (args) => {
        const m = pickModality(args.modality)
        const argv = ['identity_init', '--identity', args.identity ?? m.identity]
        if (args.creator) argv.push('--creator', String(args.creator))
        return runCore(m.core(), argv)
      },
    },
    // ---- photo ----
    {
      name: 'photo_inspect',
      description: 'Analyze a single image: dimensions, luminance (mean/std/p50/p95), saturation, mean RGB. The input for photo EDIT DECISION SCHEMA + critique. Image-only (no video modality needed).',
      parameters: {
        type: 'object',
        properties: { src: { type: 'string', description: 'Source image path.' } },
        required: ['src'],
      },
      output: outSchema,
      execute: (args) => runCore(PHOTO_CORE, ['inspect', args.src]),
    },
    {
      name: 'photo_propose',
      description: 'Propose a photo EDIT DECISION SCHEMA (crop / grade / overlay_text / resize / sharpen) from an inspect result against a rubric (intent, saturation, exposure, crop, width, caption). With group>1 it proposes a GROUP of candidates, renders and scores each one (the photo reward is defined on the rendered image), logs the group, and selects the candidate preferred by the per-creator identity when one exists.',
      parameters: {
        type: 'object',
        properties: {
          src: { type: 'string', description: 'Source image path.' },
          inspect: { type: 'string', description: 'photo_inspect JSON (stringified).' },
          rubric: { type: 'string', description: 'Rubric JSON (stringified): intent, saturation, exposure, crop, width, caption.' },
          group: { type: 'number', description: `Candidates to consider (1 = legacy single schema, no render). Default ${GROUP_DEFAULT}.` },
          identity: { type: 'string', description: 'Per-creator photo identity GGUF. Default: the workspace photo identity if it exists.' },
          select: { type: 'string', enum: ['auto', 'taste', 'objective'], description: 'auto (taste if a photo identity exists), taste, or objective.' },
          no_log: { type: 'boolean', description: 'Do not append this group to the photo training log.' },
        },
        required: ['src', 'inspect', 'rubric'],
      },
      output: outSchema,
      execute: (args) => {
        const argv = ['propose', args.src, args.inspect, args.rubric,
          '--group', String(args.group ?? GROUP_DEFAULT), '--dataset', PHOTO_DATASET]
        const identity = args.identity ?? PHOTO_IDENTITY
        if (identity && existsSync(identity)) argv.push('--identity', identity)
        if (process.env.DSH_EDITAPART_STYLE) argv.push('--style', process.env.DSH_EDITAPART_STYLE)
        if (args.select) argv.push('--select', String(args.select))
        if (args.no_log) argv.push('--no-log')
        return runCore(PHOTO_CORE, argv)
      },
    },
    {
      name: 'photo_render',
      description: 'Deterministically render a photo EDIT DECISION SCHEMA to an image with ImageMagick (crop/grade/annotate/resize/sharpen).',
      parameters: {
        type: 'object',
        properties: {
          src: { type: 'string', description: 'Source image path.' },
          schema: { type: 'string', description: 'Photo EDIT DECISION SCHEMA JSON (stringified).' },
          out: { type: 'string', description: 'Output image path.' },
        },
        required: ['src', 'schema', 'out'],
      },
      output: outSchema,
      execute: (args) => runCore(PHOTO_CORE, ['render', args.src, args.schema, args.out]),
    },
    {
      name: 'photo_critic',
      description: 'Score a rendered photo + its schema against a rubric: objective exposure/contrast/saturation metrics + per-operation keep/drop/why deltas + reward. Optionally fold a vision-model subjective score (0..1). Image-only.',
      parameters: {
        type: 'object',
        properties: {
          schema: { type: 'string', description: 'Photo EDIT DECISION SCHEMA JSON (stringified).' },
          rubric: { type: 'string', description: 'Rubric JSON (stringified): target_luma, min_contrast, width.' },
          result: { type: 'string', description: 'Rendered image path to critique.' },
          subjective: { type: 'number', description: '0..1 vision-model judgment to fold in (optional).' },
          group_id: { type: 'string', description: 'Group id from photo_propose: logs this outcome as the group\'s reward.' },
          candidate: { type: 'number', description: 'Which candidate of the group this critique is for (default 0).' },
          chosen_by: { type: 'string', enum: ['critic', 'creator', 'agent'], description: 'Who picked this candidate. A creator/agent pick is logged as a REVEALED preference and becomes the group\'s top reward.' },
        },
        required: ['schema', 'rubric', 'result'],
      },
      output: outSchema,
      execute: (args) => {
        const argv = ['critic', args.schema, args.rubric, args.result]
        if (args.subjective != null) argv.push('--subjective', String(args.subjective))
        if (args.group_id) {
          argv.push('--dataset', PHOTO_DATASET, '--group-id', String(args.group_id),
                    '--candidate', String(args.candidate ?? 0),
                    '--chosen-by', String(args.chosen_by ?? 'critic'))
        }
        return runCore(PHOTO_CORE, argv)
      },
    },
    {
      name: 'photo_revise',
      description: 'Apply a photo critic verdict to a schema: drop flagged operations and keep the rest.',
      parameters: {
        type: 'object',
        properties: {
          schema: { type: 'string', description: 'Photo EDIT DECISION SCHEMA JSON (stringified).' },
          critic: { type: 'string', description: 'Photo critic JSON (stringified) with per-op decisions.' },
        },
        required: ['schema', 'critic'],
      },
      output: outSchema,
      execute: (args) => runCore(PHOTO_CORE, ['revise', args.schema, args.critic]),
    },
  ]

  // tools.register() returns a disposer each call; the set is torn down when the
  // plugin unloads. We register into this plugin's scope layer, so they're scoped
  // to the mounted preset.
  const disposers = tools.map((t) => ctx.tools.register(t))
  const disposeAll = () => { for (const d of disposers) d() }

  // Model routing hook — guarded. Opt-in via DSH_AI_VIDEO_MODEL_ROUTE=1 (and a
  // registered video-capable provider/model). Inert by default; the session uses
  // the default model. A session's own request-header model always wins.
  if (MODEL_ROUTE_ENABLED) {
    ctx.on('agent/request', async (payload, next) => {
      const resolved = await next()
      if (payload.agent?.session?.requestHeader() !== undefined) return resolved
      return { ...resolved, provider: PROVIDER, model: MODEL }
    })
  }

  ctx.systemPrompt.section({
    name: 'tool:edit-apart',
    order: 106,
    text: 'You are a taste-driven video + photo editor (EditApart). Prefer the edit-apart tools (inventory, features, propose_schema, render_schema, review_frames, critic_edit, revise_schema, train_identity, taste_status, identity_init, and photo_inspect/photo_propose/photo_render/photo_critic/photo_revise) over ad-hoc commands for any edit task. propose_schema returns a GROUP of candidate edits and selects one; when you render a DIFFERENT candidate than the selected one, or a human picks one, log it with critic_edit chosen_by=creator|agent — that revealed pick is what teaches the per-creator taste identity. Run train_identity when the log has enough clips (taste_status shows the count), then later proposals are selected by that learned identity. Always review_frames before dropping a shot — the vision leg is normative; motion-luma is only a hint. For photos, read the result image with your eyes before dropping an operation.',
  })

  return disposeAll
}
