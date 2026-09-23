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

| Arm | Preference signal in the objective | Briefs picked differently | In the predicted direction | Mean duration (long vs short) | Training fit (long / short) |
| --- | --- | --- | --- | --- | --- |
| A | none (`pref=0`, no override) | 4 / 12 | 4 / 12 (p ≈ 0.21) | 17.54 s vs 13.90 s | 0.33 / 0.35 |
| B1 | preference loss, default weight | 8 / 12 | **8 / 12 (p ≈ 0.0006)** | 19.66 s vs 12.68 s | 0.22 / 0.26 |
| B2 | preference loss, weight ×4 | 8 / 12 | **8 / 12** | 20.12 s vs 11.97 s | 0.28 / 0.09 |
| C1 | reward override only | 8 / 12 | **8 / 12** | 21.39 s vs 11.97 s | 0.83 / 1.00 |
| C2 | override + preference loss | 8 / 12 | **8 / 12** | 21.39 s vs 11.97 s | 0.83 / 1.00 |

Chance here is not 0.5: with no taste the two identities select the *same*
candidate, so "picked differently" has a null of ~0, and the direction rate under
unrelated-but-symmetric picks is ≈ 0.20. The p-values above are one-sided
binomial tails against 0.203.

Three things worth reporting:

1. **The presence of a user-dependent term is what moves the outcome.** Without one
   the direction rate is 4/12 (p ≈ 0.21, i.e. indistinguishable from unrelated
   picks); with one it is 8/12 (p ≈ 0.0006) for every mechanism tried. That is the
   finding: taste is unidentifiable from a rubric-derived reward, and the user's
   revealed choice has to be *in the objective*.
2. **How it enters matters for the strength of the effect, not for whether it
   works.** The margins differ by a factor of ~2.5 (A +3.6 s, B1 +7.0 s, B2
   +8.2 s, C1/C2 +9.4 s), and so does the ability to fit the creator's own corpus:
   the reward override reaches 0.83/1.00 training ranking accuracy while the
   auxiliary preference loss alone stalls at 0.22–0.28. A reward that is *itself*
   made correct gives the group-relative advantage a clean signal; an auxiliary
   loss competes with a term that already points elsewhere.
3. **The reward override is still a fixed point**: C2 (override + loss) is
   indistinguishable from C1 (override alone) on every measured quantity, so the
   loss is harmless but redundant once the reward carries the preference.

> **Correction (2026-09).** An earlier version of this table reported A = 5/12,
> B1/B2 = 12/12 and C1 = 10/12, concluding that the preference loss alone was
> sufficient and the override redundant. Those numbers were produced by a GRPO
> ratio baseline that stored raw logits instead of log-probabilities, so
> `rho = 1/Z` was a per-group constant and negative advantages lost their direct
> gradient. The table above is a full re-measurement on the corrected objective
> (`tests/experiment_revealed_preference.py`); the conclusion moved from "how it
> enters does not matter" to "how it enters changes the effect size", and the
> headline claim — the user's choice must be a term in the objective — is
> unchanged.

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

**A third instance, and the point of this subsection.** The corrected table above
reversed the 1b correction itself. After the adapter could express the preference,
the ablation said "the preference loss alone suffices, the override is redundant";
with the objective defect fixed it says "every mechanism separates, but the
override produces a 2.5× larger margin and a corpus fit the loss cannot reach".
The ablation's *conclusion* changed twice, each time immediately after a defect in
the same system was fixed — first a capacity defect, then an objective/telemetry
defect. That is the pattern this subsection is about, observed twice on one
question.

**Lesson.** A "signal X is insufficient" ablation is only as trustworthy as the
model's ability to *express* X. When the mechanism under test is a gradient path,
an unidentifiable-parameter result and a vanished-gradient result are
observationally identical — both look like "the model ignored the signal". Any
such negative result should be re-run after the capacity/optimisation defects
found in the same system are fixed, and the ablation should be reported together
with the capacity conditions it was measured under. We would have published a
false claim about the learning rule had we not re-run it, and then a second false
claim had we not re-run it again after the ratio fix. **A defect found elsewhere
in the stack invalidates every ablation that touched its code path** — the
re-measurement burden is a property of the system, not of the individual
experiment.

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
| ONE unseen crop geometry | **10 / 10** | **10 / 10** | 0.300 vs 0.450 (gap 0.150 vs 0.233) |

Both identities fit their own corpora perfectly (`train_rank_acc = 1.000`). The
mechanism is a shortcut: crop geometry determines *which pixels survive*, so on
any fixed image the crop-area feature is correlated with the region statistics
(luma, contrast, salience) of the content it exposes. An adapter that can fit
both can explain the creator's picks through content statistics instead of
geometry, and those statistics shift as soon as the geometry changes. Note where
the preference can live: the shared trunk receives each group TWICE with opposite
revealed picks (the proposal is deterministic given image + rubric), so its
crop-area preference cancels and the taste can only be carried by the per-creator
latent `z_u` — the confound is a property of the adapter, not of the trunk.

Corrections to how this was first stated, from two read-only reviews and the
objective fix:

- **The `0 / 10` row was n = 1, not n = 10.** The ten evaluation briefs were ten
  *identical* briefs, so they were one distinct input evaluated ten times. The
  suite now varies the conditioning per brief (`target_luma`); on the corrected
  objective the row reads **10 / 10 over ten distinct briefs**.
- **The "no transfer" result was an artifact of a broken objective.** It was
  measured while the GRPO ratio baseline stored raw logits, so negative advantages
  had no direct gradient. With that fixed the selection transfers to the unseen
  geometry as well — with a gap of 0.150 against 0.233 in-distribution. The
  confound below therefore *narrows* the effect rather than erasing it.
- **An intermediate reading — "the ordering transfers while the selection does
  not" — is also superseded.** It was true of the pre-fix models; with the
  corrected objective both transfer on this probe (the ordering still with a
  larger relative margin). It was a real observation about a specific pair of
  models, not a property to build on.

This is the structural difference from video. In video, a group varies *which of
the available material is used* against a shared inventory, so the taste
variable (selection, duration, order) is exchanged while the content pool stays
fixed. In a static modality the group varies a *view of one image*, so taste and
content are entangled inside every group — and group-relative learning cannot
separate what the group varies from what it shares, because it normalises the
shared component away and treats the rest as signal.

**4d. Does varying the conditioning across groups repair it? Partially, and only
along the content axis — and the metric cannot resolve effects this small.**
We ran the obvious follow-up as a 2x2 (`tests/experiment_photo_transfer.py`):
{1 image, 3 images} x {narrow fixed crop family, varied crop family}, two
creators per arm, matched optimiser steps (300), one shared trunk per arm, and a
**shared distance-graded probe panel** evaluated identically for every arm
(in-family 0.267 -> near 0.168 -> mid 0.109 -> far 0.078 crop area; 3 distinct
briefs per cell; the `far` probe is below both training families' range, i.e.
extrapolation for every arm).

Behavioural result — mean selected crop-area ratio tight/loose (< 1 means the
tight-taste identity framed tighter; the in-family/seen-image cell is the
manipulation check):

| Arm (training) | in_family (seen) | near (seen) | mid (seen) | far (seen) | in_family (unseen img) | far (unseen img) |
| --- | --- | --- | --- | --- | --- | --- |
| A1 1 image, fixed | 0.222 | 0.601 | 0.400 | 0.789 | 0.555 | 1.000 |
| A2 1 image, varied | 0.222 | 1.000 | 1.000 | 0.667 | 0.222 | 1.000 |
| B1 3 images, fixed | 0.222 | 0.401 | 0.400 | 0.789 | 0.222 | **0.400** |
| B2 3 images, varied | 0.222 | 1.000 | 1.000 | 0.400 | 0.222 | 1.000 |

(ratio of mean selected crop area, tight/loose; 1.000 = no effect, < 1 = the
tight-taste identity framed tighter. Re-measured on the corrected objective with
`--no-control`, 3 distinct briefs per cell.)

Read honestly:

1. **The manipulation works**: with an in-family geometry on a trained image,
   every arm separates the two creators 3/3 at ratio 0.222.
2. **Exactly one cell shows full transfer, and it is the same one as before the
   objective fix**: B1 — three images with the *narrow* geometry family —
   separates on an **unseen image and an unseen geometry** (ratio 0.400), while
   A1, A2 and B2 show no effect there (1.000). The pre-registered hypothesis was
   that widening the geometry family would repair transfer; **that is not what
   happened.** Nor is it a content-only story any more: on a *seen* image the far
   probe separates for three of four arms (0.789/0.667/0.789/0.400), so what the
   unseen *image* costs depends on the arm.
3. **The ordering metric at this effect size is not trustworthy, and a control
   showed it.** A taste-irrelevant identity (always picks the same candidate
   index, so it carries no crop-area rule) still reaches `|tau| = 0.20–0.36` on
   crop areas. Every `far`/`mid` separation we measured (0.07–0.51) sits inside
   that band; only the in-family separations (0.62–1.49) clearly exceed it. The
   control is what stops us from reporting several "repairs" that were metric
   noise. (The control was not re-run in the post-fix pass above — the noise
   floor is a property of the metric on this data, but it is a pre-fix
   measurement and is flagged as such.)
4. **n = 3 distinct briefs per cell, one seed, one unseen image.** B1's 3/3 is
   suggestive of "content diversity forces a geometry rule", not established;
   and B1's own `far` cell on a *seen* image goes the other way (ratio 1.250),
   so the effect is not even internally consistent across image conditions.

The methodological lesson (the same shape as Finding 1b, applied to measurement):
**a positive or negative result is only as trustworthy as the demonstrated
resolution of its metric.** Here a negative control — an identity with no taste
to find — bounded the ordering metric's noise at ±0.36, which is larger than most
of the effects the experiment was built to detect. Without that control, cells
with S ≈ 0.4–0.5 and ratio ≈ 0.4 would have read as "the conditioning repaired
transfer", and the two `far` cells that actually did transfer would have been
reported alongside them as if equally solid.

Practical rules for a preparation-conditioned modality: report an identity
together with the image/brief family it was trained on; include an out-of-family
check and a **no-taste control identity** in its acceptance test; and state
whether a transfer claim is about the ordering or about the decision. Varying the
conditioning *across* groups is still the right instinct — but do it along the
axis that actually varies what the group shares (content here), and verify it
with a probe ladder rather than a single held-out point.

---

## Corrections log

Every entry is a defect found after publication of an earlier draft of these
findings, and what changed as a result. Kept in the paper deliberately: the
sequence is the methodological finding.

| # | Defect | Consequence | Fixed by |
| --- | --- | --- | --- |
| 1 | The revealed-preference claim ("only a reward override works") | Measured while the adapter could not express the preference; reversed once capacity was fixed (Finding 1b) | Finding 2's group-standardised logits + three latent injection points |
| 2 | `n = 1` presented as `n = 10` (ten identical briefs) | The photo out-of-family result overstated its evidence | Distinct `target_luma` per evaluation brief |
| 3 | **The GRPO ratio used raw logits, not log-probabilities** | `rho = 1/Z` was a per-group constant: negative advantages lost their direct gradient (identical 0.0268 for A = −0.1, −2.0, −8.0), and `clip_fraction` was a per-batch sum (5.9) | Store `log_softmax(...)`; separate the per-call and per-step denominators; `TestObjectiveFidelity` pins both |
| 4 | ImageMagick 6: `convert identify` returned 0×0 **silently** | The photo crop axis — the whole spatial feature family — was dead on IM6 while every step reported success | Resolve `identify` separately, make a dimension failure loud, refuse degenerate groups, add `selftest` |
| 5 | A render hard-required audio on every input; `loudnorm` on digital silence emits NaN | Muted screen recordings / GIF-sourced clips could not be rendered at all | Per-input audio probing + `anullsrc`; skip `loudnorm` when no input has audio |
| 6 | `"reward_obj": null` aborted the whole training log | `taste_status`/`train` failed on any log containing an unscored group | None-safe folding; unscored candidates are skipped and counted |
| 7 | The video E2E critiqued the selected schema while labelling the reward with `--candidate <pick>` | The logged reward for a revealed pick was the wrong candidate's | Candidate schemas exposed in the group metadata; the test asserts reward identity |

Findings 1, 4c and 4d were re-measured after entry 3, since all three were
computed on the code path that entry fixed. Finding 2's ablation (0/12 → 12/12)
was measured before it: both of its arms ran under the same objective, so the
*relative* conclusion stands, but its exact counts are not re-earned. Finding 3
(Muon lr) is an optimiser-scale measurement and is likewise pre-fix.

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
  10 held-out briefs, 8 candidates per group) plus the four-arm transfer
  experiment. Finding 4c's out-of-distribution row is **one geometry** (ten
  distinct briefs), so the *size* of the transfer gap is not characterised, only
  its presence on that point. Finding 4d uses 3 distinct briefs per cell, one
  seed, one unseen image, and a probe panel built by us; its positive cell (B1)
  is not internally consistent across image conditions, and the noise floor comes
  from a single control run per arm.
- Photo candidates are scored by rendering them, so the group's reward is the
  renderer's outcome; a different renderer (or an in-browser one) could move the
  thresholds the graded reward falls back on.
- Finding 1b is an argument from our own two revisions; we did not systematically
  measure how often such confounded negative ablations occur. The same caution
  applies to Finding 4d's control: one no-taste identity per arm bounds the
  metric's noise, it does not characterise its distribution.
- The creator preference is simulated throughout; no human creator was observed.
  Human taste may be more consistent (easier to learn) or less (harder).

## Reproduction

```bash
# Findings 2 and the learning numbers (no media needed, 36 tests)
python tests/test_taste_model.py -v

# Finding 1's behavioural table and Finding 2's 0/12 vs 12/12 (real footage)
DSH_EDITAPART_E2E_SRC=/path/to/footage.mp4 python tests/test_loop_e2e.py -v

# Finding 4's photo suite (features need no renderer; grouped/per-creator cells
# need ImageMagick, ~4 min)
python tests/test_photo_taste.py -v

# Finding 4d's transfer experiment (four arms x four probes x two image
# conditions; render-bound, ~25 min, writes JSON with --out)
python tests/experiment_photo_transfer.py --train-briefs 10 --eval-briefs 3 \
       --out photo_transfer.json

# Finding 1's ablation arms (train the two creators' identities per arm)
train_identity --dataset creator.jsonl --identity creator.gguf --freeze-style  # preference loss arms
train_identity --dataset creator.jsonl --identity creator.gguf --freeze-style \
               --revealed-pref-bonus 1.0 --pref-coef 0.5                       # override arms
```

Key entry points: `bin/taste_model.py` (model, objective, artifacts),
`bin/edit_apart_core.py` (`propose` group logging, `critic --chosen-by`,
`train`), `plugins/edit-apart.mjs` (the 15-tool surface).
