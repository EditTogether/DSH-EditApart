# Findings for the paper (drop-in draft)

Sections below are written to be lifted into the paper's findings/discussion with
minimal editing. Every number is reproducible from the commands cited at the end
of each finding. Nothing here is a claim about a human creator — see **Threats to
validity** at the bottom.

---

## Finding 1 — Per-user taste is unidentifiable from a rubric-derived reward; the user's revealed choice has to be a term in the objective

**Claim.** In a rubric-conditioned editing loop, a reward that is a deterministic
function of the rubric cannot identify per-user adaptation. The user's actual
choice must enter the objective. It does not matter much *how* it enters (an
auxiliary preference loss at the default weight is sufficient; overriding the
reward for that group is also sufficient but weaker alone) — what matters is that
it enters.

**Mechanism.** The critic's reward is `r = f(schema, rubric)`. It does not depend
on the user. Inside a GRPO group every candidate shares the same rubric, so the
group-relative advantage

```
A_k = (r_k − mean_j r_j) / std_j r_j
```

is a purely rubric-driven signal. Group-relative normalisation subtracts exactly
the component shared by all candidates in the group, so any user signal that was
not already inside `r` is removed rather than merely diluted. If `r ⊥ u`, then
`E[∇_{z} L]` carries no user-identifying direction and two users with opposite
tastes do not separate.

**Ablation (measured).** Two creators with opposite tastes (longest- vs
shortest-take preference over rubric-equivalent candidates), one shared
style-brain trunk, identical briefs, twelve held-out neutral briefs. The
revealed pick is logged either way; the arms differ in what the trainer does
with it:

| Arm | Preference signal in the objective | Briefs picked differently | In the predicted direction | Mean selected duration (long vs short) |
| --- | --- | --- | --- | --- |
| A | none (`pref=0`, no override) | 5 / 12 | 5 / 12 ≈ chance | 20.35 s vs 16.48 s |
| B1 | preference loss, default weight | **12 / 12** | **12 / 12** | **24.94 s vs 11.97 s** |
| B2 | preference loss, weight ×4 | 12 / 12 | 12 / 12 | 24.94 s vs 11.97 s |
| C1 | reward override only | 10 / 12 | 10 / 12 | 23.64 s vs 14.06 s |
| C2 | override + preference loss | 12 / 12 | 12 / 12 | 24.94 s vs 11.97 s |

Three things worth reporting:

1. **Without any user-dependent term the direction is at chance** (5/12), even
   though the two latents are trained on reward vectors that differ slightly
   (the logged objective reward of the candidate the user picked replaces that
   index). Logging *which* candidate the user chose therefore leaks a weak
   signal on its own — 5/12 differing — but not a usable direction. The
   "revealed choice" record is necessary; it is not sufficient.
2. **The preference loss is sufficient and saturates at a smaller weight than we
   first used** (B1 = B2 = 12/12), so the recipe is simply "put the user's choice
   in the objective", not "tune a preference weight".
3. **The reward override is also sufficient but weaker alone** (C1 = 10/12), and
   adds nothing on top of the preference loss (C2 = B1). We keep it because it is
   robust to a badly-scaled preference weight and it makes the group's top reward
   agree with the user's pick, which keeps the dense critic shaping interpretable.

**Generalisation beyond this system.** Any rubric- or instruction-conditioned RL
loop that wants per-user (or per-cohort) adaptation must ensure the objective
depends on the user's *revealed behaviour* and not only on the conditioning
input. The corollary is a logging rule: the loop has to record which output the
user actually chose, because that record is the only user-dependent variable the
objective can access.

### Finding 1b — the methodological lesson: a negative ablation can be an artifact of adapter capacity

Our first ablation of this question reached the **opposite** conclusion —
"a preference loss is not enough; the reward must be overridden". That ablation
was real, and wrong: it was measured before the adapter could express the
preference at all. At that point the trunk's scores saturated a squashing output,
the latent's gradient vanished, and the only intervention that still moved the
chosen candidate was rewriting the reward itself (which changes the *available*
policy, not just the loss). Two creators then picked identically on **12/12**
briefs unless the reward was overridden.

After fixing the adapter's capacity (Finding 2), the same ablation reverses:
the preference loss alone now separates the creators perfectly (B1), and the
override is redundant (C1 < B1). The corrected claim is the weaker, more useful
one.

**Lesson.** A "signal X is insufficient" ablation is only as trustworthy as the
model's ability to *express* X. When the mechanism under test is a gradient path,
an unidentifiable-parameter result and a vanished-gradient result are
observationally identical — both look like "the model ignored the signal". Any
such negative result should be re-run after the capacity/optimisation defects
found in the same system are fixed, and the ablation should be reported together
with the capacity conditions it was measured under. We would have published a
false claim about the learning rule had we not re-run it.

---

## Finding 2 — With a frozen shared trunk, the latent needs an output path (saturation silently deletes it)

**Claim.** Per-creator adaptation against a frozen shared trunk fails if the
trunk's score saturates — and it fails *silently*: the latent learns the correct
direction and still changes nothing about the decisions.

**Mechanism.** Under a group-relative margin objective the trunk drives its
scores to the extremes of a squashing nonlinearity (`tanh` → ±1). Once
`|pre-activation|` is large, `(1 − u²) → 0`, so the gradient the latent receives
vanishes and its contribution to the *ordering* is annihilated. Measured on one
trained pair: the two latents differed by `+0.48` vs `−0.84` on the same feature
(A's derivative of utility w.r.t. duration was positive, B's negative) and both
identities still selected the identical candidate on 12/12 briefs.

**Fix.** Standardise the utility *within each candidate group* before the softmax
(`s_k = ((u_k − mean u)/std u)/T`), and give the latent a direct output path
(FiLM on the hidden layer **plus** an output gate **plus** a per-creator linear
readout over the features). The group standardisation keeps logits responsive by
construction, so the latent can always reorder candidates.

| Variant | Neutral briefs differing between the two identities |
| --- | --- |
| Squashed output, hidden-layer FiLM only | 0 / 12 |
| Group-standardised logits + three latent injection points | **12 / 12** |

The generalisable lesson: **a per-user adapter needs a monotone, non-saturating
path to the decision variable.** If the adapter modulates an intermediate
representation whose output is squashed, the adapter's authority is not merely
reduced — it is asymptotically zero, and the failure presents as "the identity
learned something (its weights moved, its sign is right) yet the system's
behaviour is unchanged."

---

## Finding 3 — Muon's learning rate is in spectral-norm units, which bites on small matrices

**Claim.** Muon's step is normalised to unit spectral norm, so its effective step
size is independent of gradient magnitude. Reusing a typical Adam learning rate
therefore oversteps on small weight matrices and the model stops fitting.

**Measured.** Same objective, same data, 2D weights on Muon / 1D on AdamW:

| Muon lr | Training ranking accuracy |
| --- | --- |
| 0.02 | ≈ 0.50 (stalls — looks like "the objective does not learn") |
| 0.005 | ≈ 0.85 |

This is the second instance of the Finding 1b pattern *inside our own stack*: a
plateau that reads as "the objective does not learn" was an optimiser-scale
defect. Both were diagnosed by asking "can the model express the thing at all?"
before blaming the learning rule.

**Recipe.** Treat Muon's lr as a different unit: on a 16×24 trunk it belongs
~10× below the Adam lr used for the latent and biases, and the failure mode at
too-high lr is a plateau rather than divergence. Report a learning-rate
sensitivity rather than a single number.

---

## Finding 4 — Static-modality taste: synthesise the missing axis, grade the reward, and do not confound taste with content

Photo (single-image) editing is the harder modality, and for reasons that are
structural rather than incidental. Three separate effects were measured on the
wired photo loop.

**4a. With no temporal axis, the spatial structure must be synthesised.**
Video's layout reads the shot inventory: pacing, shot selection, order, energy.
A single image has none of that, so the feature layout has to invent spatial
structure — the crop's geometry (area, aspect deviation, centre deviation,
distance from the rule-of-thirds points) plus statistics of the *region the crop
keeps*, taken from a 4×4 region grid computed from the same downscale the global
inspect already used. Without the region features a crop-preferring creator is
indistinguishable from a grade-preferring one, because nothing in the vector says
*where* the crop lands or *what* it keeps.

**4b. A quantised reward makes candidate groups tie, which is exactly zero
group-relative signal.** The photo critic scored exposure as a threshold
(`|mean luma − target| ≤ 0.15`). Measured on a 6-candidate group: all six scored
−0.05, `spread = 0.0`, i.e. the GRPO advantage `A_k = (r_k − mean r)/std r` is
identically zero for every candidate. This is a reward-design defect that is
invisible in single-candidate use and fatal for group-relative learning. Grading
the term — full credit inside the tolerance exactly as before, a linear falloff
outside it — raised the same group to `spread = 0.47`, and 11/18 training groups
then carried non-zero spread. The lesson generalises: **before blaming the
learned policy, check that the reward actually varies across the group.**

**4c. The taste variable is confounded with the content it reveals, so the
learned preference does not transfer to a new geometry.** This is the finding we
did not expect. Two creators with opposite framing taste (tightest vs loosest
crop) were trained on one image across 18 briefs, sharing one frozen trunk:

| Evaluation briefs | Briefs picked differently | In the predicted direction | Mean selected crop area (tight vs loose) |
| --- | --- | --- | --- |
| Held-out, same brief/crop family | **10 / 10** | **10 / 10** | **0.155 vs 0.389** |
| Unseen crop geometry | 0 / 10 | 0 / 10 | 0.450 vs 0.450 (identical) |

Both identities fit their own corpora perfectly (`train_rank_acc = 1.000`) and
learned the correct sign on the `crop_area` feature weight (−0.22 vs +0.15), yet
on a new geometry both picked the same candidate. The mechanism is a shortcut:
crop geometry determines *which pixels survive*, so on any fixed image the
crop-area feature is correlated with the region statistics (luma, contrast,
salience) of the content it exposes. A model that can fit both can explain the
creator's picks through content statistics instead of geometry, and those
statistics shift as soon as the geometry changes. Training on three images
instead of one did **not** restore the transfer in our measurement.

This is the structural difference from video. In video, a group varies *which of
the available material is used* against a shared inventory, so the taste
variable (selection, duration, order) is exchanged while the content pool stays
fixed. In a static modality the group varies a *view of one image*, so taste and
content are entangled inside every group — and group-relative learning cannot
separate what the group varies from what it shares, because it normalises the
shared component away and treats the rest as signal.

Practical rule for a preparation-conditioned modality: vary the *conditioning*
across groups (many images, many crop geometries), not only the knobs within a
group; otherwise the learned preference is an image-specific shortcut. A
per-creator photo identity should be reported together with the image/brief
family it was trained on, and an out-of-family check should be part of its
acceptance test.

---

## Threats to validity

- The per-creator preference in the ablations is **simulated** by a deterministic
  policy over candidates (pick the longest / shortest edit). A real creator's
  clicks may be noisier, and the human-in-the-loop variant is untested; the claim
  established here is about the *objective's identifiability*, not about human
  fidelity.
- The behavioural numbers come from one clip (41.7 s, 23 shots) across twelve
  neutral briefs, with 25 training briefs per creator. The learning numbers use
  synthetic hidden preferences across three seeds. Twelve briefs is enough for a
  sign test (12/12, p ≈ 2⁻¹²) but the *effect sizes* are single-clip.
- The trained policy is the **candidate selector**, not a fine-tuned language
  model: it can only choose among the candidates the engine proposes, so "taste"
  here means "ranking over a fixed reference set", not generation.
- The photo findings come from one synthetic 1600×1200 image (18 training briefs,
  10 held-out briefs, 8 candidates per group) plus one three-image run; the
  out-of-distribution result is a single geometry, so the *size* of the transfer
  gap is not characterised, only its presence.
- Photo candidates are scored by rendering them, so the group's reward is the
  renderer's outcome; a different renderer (or an in-browser one) could move the
  thresholds the graded reward falls back on.
- Finding 1b is an argument from our own two revisions; we did not systematically
  measure how often such confounded negative ablations occur.

## Reproduction

```bash
# Findings 2 and the learning numbers (no media needed, 36 tests)
python tests/test_taste_model.py -v

# Finding 1's behavioural table and Finding 2's 0/12 vs 12/12 (real footage)
DSH_EDITAPART_E2E_SRC=/path/to/footage.mp4 python tests/test_loop_e2e.py -v

# Finding 1's ablation arms (train the two creators' identities per arm)
train_identity --dataset creator.jsonl --identity creator.gguf --freeze-style  # preference loss arms
train_identity --dataset creator.jsonl --identity creator.gguf --freeze-style \
               --revealed-pref-bonus 1.0 --pref-coef 0.5                       # override arms
```

Key entry points: `bin/taste_model.py` (model, objective, artifacts),
`bin/edit_apart_core.py` (`propose` group logging, `critic --chosen-by`,
`train`), `plugins/edit-apart.mjs` (the 15-tool surface).
