# A 3-month playbook for this competition (and segmentation Kaggle competitions generally)

Written for someone new to Kaggle, grounded in what actually cost time and what actually
paid off while working this specific competition (solar filament instance segmentation,
scored by Panoptic Quality). Where a lesson is general, it's flagged as such; where it's
specific to this dataset/metric, that's flagged too.

The single organizing principle: **cheap, certain wins before expensive, uncertain
ones.** Postprocessing tuning before retraining. Retraining before architecture search.
Architecture search before novel loss functions. Diagnosis before prescription, always.

---

## Month 1 (weeks 1-4): Get a real, ugly, end-to-end submission fast

### Week 1: Understand the problem before writing any model code

- **Reimplement the competition's metric locally, exactly.** Not an approximation --
  the actual formula. A mismatch between your local validation number and the
  leaderboard is one of the most time-wasting things that can happen mid-competition,
  because every decision you make based on local numbers is then untrustworthy. (Here:
  `src/utils/panoptic.py`'s `panoptic_quality()` matches the Overview page's PQ formula
  exactly, matched via Hungarian assignment at IoU > 0.5.)
- **Do real EDA before modeling**: annotation format and quirks, image resolution,
  instrument/capture variety (this dataset has multiple telescope codes baked into
  filenames), the distribution of instance count and size per image. You will use this
  later -- e.g. we found failure modes correlate with instance count (crowded images
  under-segment) only because we'd already looked at the distribution.
- **Understand what's actually being scored.** This competition scores *instances*
  (Panoptic Quality), not just pixel accuracy -- meaning postprocessing (turning one
  probability map into separate instances) is not an afterthought, it's a first-class
  part of the problem, as important as the model itself. Realizing this early would
  have saved real time; we discovered it mid-competition.

### Weeks 1-2: Ship the simplest possible full pipeline

Build data loading -> tiny/fast model -> training loop -> prediction -> postprocessing
-> submission file, end to end, before caring at all about score. The goal is
validating that every stage's *interface* is correct (shapes, formats, coordinate
systems, the exact submission schema) while the stakes are low. Get a submission
uploaded on day one or two, even a bad one -- it's your first real leaderboard signal
and it forces the whole pipeline to actually work.

### Weeks 3-4: Real baseline + infrastructure that will pay for itself

Train a real (if modest) model: standard encoder-decoder segmentation architecture, a
well-known pretrained encoder, standard augmentation, a sane starting resolution (not
necessarily native -- see the resolution/VRAM notes below). Get semantic *and*
instance-level metrics computed automatically, not eyeballed.

Set up infrastructure now, not later -- every item below was worth its cost this
session:
- **An experiment log** (see this repo's `memory.md`): one file, append-only, one
  section per experiment with what you tried, why, and what happened. Do this from day
  one; we started it mid-competition and immediately wished we'd started sooner.
  Recording *why* something was tried, not just the result, is what let us later reason
  about a counterintuitive result (a loss change that regressed the exact metric it was
  meant to improve) as a real finding instead of shrugging it off as noise.
- **Auto-named, non-clobbering run outputs.** A fixed `checkpoint_dir` will silently
  overwrite your best checkpoint the moment you start a second experiment. Give every
  run its own folder (timestamp + key config values) from the start.
- **Resume support**, and get it *right*: naive resume that doesn't also restore
  optimizer/scheduler state will silently distort your LR schedule and Adam momentum on
  every resume. Test it by actually resuming and checking the logged LR curve
  continues smoothly, not just that training doesn't crash.
- **Multi-GPU, if you have it, working correctly from the start.** This is the
  single most time-consuming infrastructure item if you delay it: we hit a DDP deadlock
  from `DistributedDataParallel`'s automatic buffer-broadcast colliding with a
  validation loop that only ran on one rank, and a separate OOM caused by a validation
  batch size that wasn't actually divided across ranks. Both are the kind of bug that's
  cheap to catch on day 3 and expensive to catch on day 30 when it silently hangs a run
  for 10 minutes before timing out.

**Month 1 deliverable checklist**: a submitted (if mediocre) entry; a locally
reproducible metric that matches the leaderboard's logic; an experiment log with at
least one entry; a training script that can resume correctly and doesn't overwrite
prior runs.

---

## Month 2 (weeks 5-8): Disciplined, evidence-driven iteration

This is where most time gets wasted on a first Kaggle competition, by guessing at fixes
instead of measuring what's actually wrong. The order below is roughly cheapest-and-most-certain-first --
don't skip ahead to the expensive stuff before ruling out the cheap stuff.

### Step 1: Postprocessing, before touching the model at all

If your task has any postprocessing step (thresholding, instance-splitting,
NMS, whatever) between the model's raw output and the scored prediction, **tune that
first.** It requires no retraining, iterates in seconds to minutes instead of hours,
and can be a bigger lever than the model itself. In this competition, tuning a single
watershed parameter took Panoptic Quality from 0.039 to 0.34 -- an 8x improvement,
purely from postprocessing, on a model we'd already trained and moved past.

Build a *sweep* tool for this, not a manual one-value-at-a-time loop: predict once,
cache the raw model outputs, then cheaply re-run just the postprocessing step across a
grid of parameter values (see this repo's `evaluate.py --sweep-watershed-min-distance`).
The GPU forward pass is almost always the expensive part -- don't pay for it repeatedly
just to test a CPU-side parameter.

### Step 2: Diagnose before prescribing

Before deciding *what* to change about the model, look at the actual training curves.
Plot train loss and val loss together. Ask:
- Are they diverging (val stuck or rising while train keeps dropping)? -> overfitting.
  Fix: augmentation, regularization, more data -- not more model capacity.
- Are they converging together and then both plateauing, even as the LR schedule fully
  decays with no further improvement? -> a genuine capacity/information ceiling
  (underfitting), not a training-duration problem. More epochs will not help; this is
  what we found and it ruled out "just train longer" immediately.

This five-minute check saves you from the single most common wasted experiment: adding
more epochs (or more augmentation, if it's actually underfitting) to a model that's
already stopped learning for a structural reason.

### Step 3: Build error-analysis tooling before picking a fix

A single scalar metric (Dice, IoU, PQ, mAP) tells you *that* something is wrong, not
*why*. Before choosing a fix, build tooling that breaks the failure down:
- **Per-image metrics**, not just an aggregate mean -- computed once, saved to a CSV,
  so you can sort/filter/histogram them yourself instead of re-running the model every
  time you have a new question.
- **Automatically visualize the worst-performing examples**, not random spot-checks.
  Random samples mostly show you what's already working.
- **Split any composite metric into its directional components.** Dice conflates
  precision and recall into one number; splitting them tells you whether the model is
  under-predicting (cautious, misses things) or over-predicting (trigger-happy, invents
  things) -- two problems with opposite fixes.
- **Characterize false positives and false negatives by shape/size**, not just count
  them. We found ours were nearly indistinguishable from each other (same size
  distribution, both elongated, not blob-like) -- a finding that ruled out "just filter
  small blobs in postprocessing" and pointed instead at a genuine model discrimination
  problem, which we'd never have found from the aggregate PQ number alone.

This tooling is a real time investment (multiple dedicated scripts, in our case), but
it's what turns "the model still doesn't work well, let's try something" into a
specific, falsifiable hypothesis about *why*, which is the only way to spend the rest
of your model-improvement budget efficiently.

### Step 4: Change one variable at a time

If you're testing whether more resolution helps and whether more model capacity helps,
test them *separately* before testing them together. We deliberately isolated a bigger
encoder from a resolution increase specifically so a result could be attributed to one
or the other -- then combined them once each was independently understood. Skipping
this is how you end up with an improved (or regressed) number and no idea which of your
three simultaneous changes caused it.

Practical corollary: **resolution and model capacity both cost VRAM, and they trade off
against each other.** If you're memory-constrained, know your levers for buying back
headroom (gradient checkpointing trades compute time for activation memory; smaller
batch size; a lighter encoder) *before* you need them, and re-tune batch size
empirically every time you change resolution, encoder, or precision settings -- what
fit before will not necessarily fit after.

### Step 5: Only now, architecture and loss changes

Encoder swaps, decoder architecture changes, custom loss terms -- each justified by a
specific finding from step 3, not by "this is what other people use." A loss change
motivated by a concrete measurement (e.g. "matched-instance overlap is consistently
loose even for correct detections, so add a term that directly optimizes overlap") is
falsifiable: you know exactly which number to check afterward to see if it worked. A
loss change picked because it's popular is not -- you won't know what to conclude from
the result either way.

**Month 2 deliverable checklist**: a postprocessing-tuning tool and its tuned values
committed; a documented diagnosis of over/underfitting from real training curves;
per-image error-analysis tooling with at least one worst-case visualization pass; at
least one cleanly isolated (single-variable) experiment with a logged conclusion.

---

## Month 3 (weeks 9-12): Squeeze and de-risk

- **Cross-validation, not a single fixed split**, once you're optimizing for small
  differences. A single held-out split's noise matters more than you'd think on a
  small dataset -- 1039 training images here, and a single-split PQ can move by more
  than some of the improvements you'll be chasing.
- **Test-time augmentation** (average predictions over flips/rotations at inference) --
  cheap, no retraining required, usually a small reliable boost.
- **Ensemble your diverse checkpoints.** If you followed Month 2's advice, you already
  have several independently-trained models from isolated experiments (different
  resolutions, encoders, loss configs) sitting around -- averaging their predictions is
  close to free at this point and typically beats any single one of them.
- **Re-sweep postprocessing on the final model/ensemble.** The optimal postprocessing
  parameters are tied to a specific model's output characteristics -- we found the
  tuned value shifted meaningfully between a 768px and a 1536px model of the same
  architecture. Don't assume an earlier sweep still applies to your final model.
- **Reserve real time at the end purely for submission validation**: format edge cases,
  off-by-one coordinate issues, empty-prediction handling, the works. Never let this be
  a day-of-deadline discovery.

**Month 3 deliverable checklist**: a cross-validated model selection process; a TTA
pass; an ensemble of at least 2-3 diverse models; a final postprocessing sweep on the
ensemble's actual output; a submission validated well before the deadline.

---

## Cross-cutting habits that mattered more than any single technique

1. **Write down why, not just what.** An experiment log that only records outcomes is
   much less useful than one that records the hypothesis behind each attempt --
   especially for surprising results, which are exactly the ones worth remembering.
2. **Isolate variables religiously.** It's tempting to bundle changes to save wall-clock
   time; it almost always costs more time later in confused attribution.
3. **Build small tools instead of doing things by hand repeatedly.** A sweep script, a
   per-image metrics CSV, an auto-flagging worst-case visualizer -- each took real time
   to build and paid for itself the second time it was used, let alone the fifth.
4. **Prefer diagnosis you can point to over intuition.** "I think the model needs more
   capacity" is a guess. "Train and val loss converged and plateaued together even as
   LR fully decayed" is evidence. Chase the second kind.
5. **Expect infrastructure bugs on constrained/multi-GPU hardware to eat real time**,
   and budget for it rather than being surprised by it. They are often silent (a hang,
   not a crash) and specifically show up under conditions you haven't hit yet (a
   different batch size, a resumed run, a new resolution) -- which is exactly why
   isolating variables and testing infrastructure changes in isolation (same principle
   as Step 4 above, applied to your own tooling) matters just as much as it does for
   modeling choices.
