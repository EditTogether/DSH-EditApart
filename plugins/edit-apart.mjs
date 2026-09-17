// EditApart — preset-owning tool plugin for the EditTogether community.
//
// A single import-free plugin mounting ALL of the preset's editing capability:
//   * 7 video tools  (inventory, features, propose_schema, render_schema,
//                     review_frames, critic_edit, revise_schema) -> bin/edit_apart_core.py
//   * 5 photo tools  (photo_inspect, photo_propose, photo_render, photo_critic,
//                     photo_revise)                           -> bin/photo_core.py
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
import { readFileSync } from 'node:fs'

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
      description: 'Propose an EDIT DECISION SCHEMA from a shot inventory against a rubric (pacing + shot-selection; v0).',
      parameters: {
        type: 'object',
        properties: {
          inventory: { type: 'string', description: 'Inventory JSON (stringified).' },
          rubric: { type: 'string', description: 'Rubric JSON (stringified): min_shot_dur, max_shot_dur, target_duration, pace.' },
        },
        required: ['inventory', 'rubric'],
      },
      output: outSchema,
      execute: (args) => runCore(VIDEO_CORE, ['propose', args.inventory, args.rubric]),
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
        },
        required: ['schema', 'inventory', 'rubric'],
      },
      output: outSchema,
      execute: (args) => {
        const sub = args.subjective == null ? [] : ['--subjective', String(args.subjective)]
        return runCore(VIDEO_CORE, ['critic', args.schema, args.inventory, args.rubric, ...sub])
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
      description: 'Propose a photo EDIT DECISION SCHEMA (crop / grade / overlay_text / resize / sharpen) from an inspect result against a rubric (intent, saturation, exposure, crop, width, caption).',
      parameters: {
        type: 'object',
        properties: {
          src: { type: 'string', description: 'Source image path.' },
          inspect: { type: 'string', description: 'photo_inspect JSON (stringified).' },
          rubric: { type: 'string', description: 'Rubric JSON (stringified): intent, saturation, exposure, crop, width, caption.' },
        },
        required: ['src', 'inspect', 'rubric'],
      },
      output: outSchema,
      execute: (args) => runCore(PHOTO_CORE, ['propose', args.src, args.inspect, args.rubric]),
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
        },
        required: ['schema', 'rubric', 'result'],
      },
      output: outSchema,
      execute: (args) => {
        const sub = args.subjective == null ? [] : ['--subjective', String(args.subjective)]
        return runCore(PHOTO_CORE, ['critic', args.schema, args.rubric, args.result, ...sub])
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
    text: 'You are a taste-driven video + photo editor (EditApart). Prefer the edit-apart tools (inventory, features, propose_schema, render_schema, review_frames, critic_edit, revise_schema, and photo_inspect/photo_propose/photo_render/photo_critic/photo_revise) over ad-hoc commands for any edit task. Always review_frames before dropping a shot — the vision leg is normative; motion-luma is only a hint. For photos, read the result image with your eyes before dropping an operation.',
  })

  return disposeAll
}
