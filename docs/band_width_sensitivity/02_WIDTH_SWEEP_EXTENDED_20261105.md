# Extended band-width sweep: does the trend continue past 0.30?

**Answer in one line.** The trend does **not** continue: width 0.50 turns over
on every metric, so the optimum is interior, sits at 0.20-0.30, and is now
bracketed on both sides. But the reason is not the one §F expected — sparsity is *best* at
0.50, so the decline is information loss, not estimation noise.

§F was written before the fits finished and is left exactly as it stood, so its
predictions can be scored against what happened. §G and §H were written
afterwards.

---

## A. Why this run exists

The four-width study
(`docs/band_width_sensitivity/01_BAND_WIDTH_SENSITIVITY_20261105.md`) swept
0.05, 0.10, 0.20 and 0.30 and found the metrics improving monotonically as the
bands got **wider**: AUC-ROC 0.8739 → 0.8758 → 0.8775 → 0.8784, Brier 0.1227 →
0.1221 → 0.1209 → 0.1199, with the parameter count falling from 224 to 77 along
the way. Width 0.30 was the best banded configuration on five of six metrics and
also the simplest.

The problem is that 0.30 was the edge of the tested range and the trend had not
turned over. That left two readings the data could not separate:

1. **There is an interior optimum above 0.30** that the sweep did not reach.
2. **There is no interior optimum**, and the trend simply runs toward the
   degenerate limit — the metrics improve because the main-range bands are
   getting less harmful, not more informative.

This run adds a fifth width, 0.50, to separate them. It is a post-hoc extension
of a four-width design, not part of the pre-specified sweep, and is reported as
one.

## B. What changed, and what did not

Only the list of widths. The band-width list is now a `--widths` command-line
argument whose **default is still the pre-specified four**, so running the script
with no arguments reproduces the original design; 0.50 has to be asked for
explicitly. The driver, the model, the cohort, the folds and the metrics are
untouched, and the four-width run was re-verified after the change: resuming it
reproduces all five arms' metrics and every contrast, with 21/21 checks passing.

This run re-fits every arm from scratch rather than bolting 0.50 onto the
earlier predictions. Two reasons: the checkpoint is keyed on the held-out
patient rather than on the arm, so a resume would have skipped all 934 folds and
never fitted the new arm; and re-fitting means every contrast in this report
comes from one internally consistent set of predictions, with no cross-run
merging to justify. **The earlier run is not overwritten** — this is a new
directory.

## C. The width-0.50 configuration

3.0 ÷ 0.5 = 6 exactly, so the width tiles the main range with no remainder.
Bands are left-closed and right-open, as in every other configuration, and the
edges are computed as `k · span / n` so they are exact in floating point.

Sixteen bands — six main, ten tail — carry twenty joint categories:

| band | subtypes that can occupy it |
|---|---|
| [0.00, 0.50) | PIGD |
| [0.50, 1.00) | Indeterminate, **PIGD** |
| [1.00, 1.50) | **TD**, Indeterminate |
| [1.50, 2.00) … [2.50, 3.00) | TD |
| ten tail bands, [3.00, 3.25) … [7.50, ∞) | TD |

Two bands are straddled by a subtype threshold — 0.90 cuts [0.50, 1.00) and 1.15
cuts [1.00, 1.50) — which is the same count as width 0.20. Neither threshold
lands on the 0.50 grid, so this width does not introduce a new variant of the
straddling confound described in the four-width report; it sits with 0.20.

The twenty categories are 15 TD (four main bands, ten tail bands, and one for
undefined ratios), 3 Indeterminate (two main bands plus undefined) and 2 PIGD
(two main bands). There is no PIGD-undefined category because an undefined ratio
means `pigd_score == 0`, which cannot be a PIGD visit. The 692 undefined ratios
split 634 TD and 58 Indeterminate.

**The tail is identical to every other configuration**: the published rule,
34 candidate 0.25-wide bands from 3.0 merged upward until each holds ≥ 30
observations from ≥ 20 training patients, yielding the same ten final bands with
lows `[3.00, 3.25, 3.75, 4.25, 4.75, 5.25, 5.50, 6.00, 6.50, 7.50]`. Only the
main-range width varies across arms.

## D. Complexity and sparsity

Over the whole cohort's 4,295 labelled visits:

| width | main bands | categories | params | empty | <15 obs | <30 obs | smallest non-empty | median | obs/param |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 60 | 73 | 224 | 11 | 32 | 40 | 1 | 22 | 19.2 |
| 0.10 | 30 | 44 | 137 | 1 | 9 | 12 | 1 | 71 | 31.4 |
| 0.20 | 15 | 29 | 92 | 0 | 1 | 2 | 8 | 135 | 46.7 |
| 0.30 | 10 | 24 | 77 | 1 | 2 | 2 | 8 | 137 | 55.8 |
| **0.50** | **6** | **20** | **65** | **0** | **0** | **0** | **35** | **148** | **66.1** |

Width 0.50 is the first configuration in which **every category clears 30
observations** — the same bar the tail-merging rule itself enforces. Its
sparsest category holds 35 observations from 33 patients, against 8 at widths
0.20 and 0.30 and 1 at 0.05 and 0.10.

This matters for interpreting the result, because the leading explanation for
the four-width trend was that wider bands were winning by escaping sparsity. At
0.50 there is no sparsity left to escape. If the metrics keep improving here,
sparsity was not the whole story; if they stop improving, it largely was.

## E. What is held fixed

Identical to the four-width study and asserted per fold rather than assumed:
three hidden states, the continuous-time generator `Q`, interval-aware
`exp(Q·dt)`, forward filtering, the 934-patient cohort, 3,361 cases, 3,029
eligible on one shared denominator for all six arms, Indeterminate kept as input
evidence and never scored as a target, the emission
`E[a,(s,b)] = M[a,s] × C[a,s,b]` with the 0.5 floor, seed 42, four restarts.
The panel is the frozen one (md5 `fb038ddbdc19ab3368e13940f398e7ba`, verified at
run start).

No held-out patient influences band construction, tail merging, fitting,
category selection or the cohort: bands are built from training patients only,
per fold, and the driver asserts this rather than trusting it. Width is not
tuned against held-out performance inside the fits.

## F. What each outcome will mean — stated before the results

The two readings in §A make different predictions, and the useful thing is that
the metrics disagree with each other in a diagnostic way. In the four-width run
the unbanded reference was **better** than every banded arm on accuracy (0.8369)
and calibration (0.0065) and **worse** on AUC (0.8587 against 0.8784 at width
0.30). So:

- **If 0.50 keeps improving on AUC-ROC, PR-AUC and Brier**, the trend is not
  mere convergence toward the unbanded model — AUC is moving *away* from the
  unbanded value, not toward it. That favours reading 1, a real optimum at or
  beyond 0.50, and makes a further width worth running.
- **If 0.50 flattens or turns over**, the four-width trend was the sparsity
  effect running out, and 0.30 is at or near the best this family can do. That
  favours reading 2.
- **If 0.50's accuracy and calibration improve while its AUC falls**, that is
  the signature of convergence toward the unbanded model, and the clearest
  possible support for reading 2.

One nuance worth stating now, because it changes what "the degenerate limit"
means. Widening the main range does **not** converge on the subtype-only
reference: at the limit of one main band the model still carries the ten tail
bands, so it becomes *subtype plus a tail indicator*, not *subtype alone*. If
the metrics improve all the way to that limit, the implication is that the
useful ratio information lives in the **tail** — the extreme-TD region above 3.0
— and that main-range resolution was the harmful part all along. That is a
sharper and more testable claim than "wider is better", and a single run at the
degenerate width would settle it.

Whatever the outcome, two cautions from the four-width study carry over
unchanged and will not be relaxed here: comparing configurations on held-out
data and naming a winner is selection on the evaluation set, so the winner's
numbers are optimistically biased; and AUC gains concentrated above FPR 0.2 are
not clinically usable, so partial AUC in the operating region is the number that
decides whether anything here matters in practice.

## G. Results

The run completed in 1h58m: 934 folds x 6 arms = **5,604 fits, every one
converged, no fold failed**, 24/24 validity checks pass.

**The four earlier arms reproduce the four-width run exactly** — maximum
absolute difference 0.000e+00 across accuracy, AUC, PR-AUC, partial AUC, Brier
and log loss, for all five previously-fitted arms. Adding a sixth arm changed
nothing about the others, which is both a clean independent replication and a
direct demonstration that each arm's fit is independent of which other arms are
in the run.

### The sweep turns over

| width | accuracy | AUC-ROC | PR-AUC | partial AUC | Brier | log loss | calib. gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| subtype only | **0.8369** | 0.8587 | 0.7496 | 0.7883 | 0.1214 | 0.3961 | **0.0065** |
| 0.05 | 0.8336 | 0.8739 | 0.7330 | 0.7802 | 0.1227 | 0.3952 | 0.0246 |
| 0.10 | 0.8316 | 0.8758 | 0.7398 | 0.7823 | 0.1221 | 0.3930 | 0.0283 |
| 0.20 | 0.8336 | 0.8775 | **0.7540** | 0.7868 | 0.1209 | 0.3898 | 0.0238 |
| **0.30** | 0.8359 | **0.8784** | 0.7523 | **0.7892** | **0.1199** | **0.3874** | 0.0222 |
| 0.50 | 0.8320 | 0.8765 | 0.7409 | 0.7827 | 0.1217 | 0.3916 | 0.0289 |

Six of the seven metrics peak at **0.30** — accuracy, AUC-ROC, partial AUC,
Brier, log loss and calibration — and PR-AUC peaks one step earlier, at 0.20.
Whichever of the two is taken as the summit, **every metric falls back at
0.50**, and the reversal is not selective: all seven move the wrong way
together.

### Width 0.50 gives up the whole of 0.30's advantage

Paired patient bootstrap against the 0.10 baseline, 2,000 resamples:

| arm | AUC Δ | 95% CI | PR-AUC Δ | partial AUC Δ | 95% CI |
|---|---:|---|---:|---:|---|
| 0.20 | +0.0017 | [−0.0002, +0.0036] | **+0.0142** | **+0.0045** | **[+0.0010, +0.0082]** |
| **0.30** | **+0.0026** | **[+0.0001, +0.0051]** | **+0.0125** | **+0.0068** | **[+0.0022, +0.0118]** |
| 0.50 | +0.0006 | [−0.0023, +0.0036] | +0.0011 | +0.0003 | [−0.0066, +0.0068] |

Width 0.30 beat the baseline on all three with intervals excluding zero. Width
0.50 is **indistinguishable from the baseline on every one of them**. On
calibration it is worse than the baseline (+0.0006), the only arm in the study
of which that is true, making it at 0.0289 the **worst-calibrated arm overall**
— 4.5x the unbanded reference.

Accuracy, as in the four-width study, separates nothing: every interval includes
zero, and 0.50's margin over the baseline is +0.03 pp with McNemar p = 1.00.

### Where it breaks: PIGD, not TD

AUC within each current-label stratum:

| stratum | n | unbanded | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TD | 2,080 | 0.6693 | 0.7442 | 0.7501 | 0.7503 | 0.7511 | **0.7528** |
| Indeterminate | 270 | 0.6328 | 0.6369 | 0.6348 | **0.6379** | 0.6326 | **0.6379** |
| PIGD | 679 | 0.6649 | 0.6645 | 0.6728 | **0.6898** | 0.6857 | **0.6580** |

The two strata move in opposite directions. TD discrimination keeps improving
all the way to 0.50, where it is best of all six arms. PIGD discrimination
collapses at 0.50 to **0.6580 — below even the unbanded reference**, and the
worst of any arm.

The alphabet says why. The number of categories each subtype can occupy:

| width | TD | Indeterminate | PIGD |
|---|---:|---:|---:|
| 0.05 | 48 | 6 | 19 |
| 0.10 | 30 | 4 | 10 |
| 0.20 | 21 | 3 | 5 |
| 0.30 | 18 | 2 | 4 |
| 0.50 | 15 | 3 | **2** |

TD owns the ten tail bands, which are identical at every width, so widening the
main range barely touches it — 18 categories at 0.30, still 15 at 0.50. PIGD
lives entirely below ratio 1.0, so it is cut from 4 categories to **2** by the
move from 0.30 to 0.50. The model loses the ability to tell a ratio of 0.1 from
one of 0.4, and the positive class of the prediction target is exactly the class
whose resolution was destroyed.

## H. Interpretation

**The question is answered: the optimum is interior and now bracketed on both
sides.** Reading 1 from §A is correct in form — there is an optimum — but it
sits at 0.20-0.30 rather than beyond 0.30, and the four-width sweep had already
reached it without being able to prove it was a peak. Six metrics put the summit
at 0.30 and PR-AUC puts it at 0.20; the sweep cannot separate those two.

**§F predicted the shape and got the mechanism wrong, which is worth recording.**
The written prediction was that a turn-over would mean "the four-width trend was
the sparsity effect running out". The turn-over happened; the explanation does
not survive contact with the data. Width 0.50 is the *least* sparse
configuration in the study — zero empty categories, zero below 30 observations,
every category at 35 observations or more, 66 observations per parameter. If
sparsity were the active variable, 0.50 should have been the best arm. It was
among the worst. So the rising limb from 0.05 to 0.30 cannot be attributed to
sparsity alone either; something was improving that was not estimation noise.

**What the sweep is actually trading off is resolution against blur, and the two
subtypes want opposite things.** The TD signal is concentrated in the tail above
ratio 3.0, which no main-range width touches, so TD only benefits as main-range
noise is stripped away — monotone right out to 0.50. The PIGD signal is
concentrated below 1.0, in a stretch of the main range just 1.0 wide, and that
is precisely what coarsening destroys: 4 categories at 0.30, 2 at 0.50. The
overall optimum at 0.30 is not a property of the ratio in general; it is the
point where TD's preference for coarser bands and PIGD's need for finer ones
balance, on this cohort, for this target.

This also retires the "degenerate limit" conjecture raised in §F. The idea that
metrics might improve all the way to a single main band — implying the useful
information is entirely in the tail — is now ruled out: performance turns down
well before that limit, and it turns down because of PIGD. The tail does carry
TD's information, as that conjecture supposed, but a tail-only model would
discard the PIGD signal altogether.

**None of this changes the practical conclusion, and it is important not to
oversell the finding.** The optimum is real but small: 0.30's advantage over the
0.10 baseline is +0.0026 AUC. No width at any point in the sweep improves
next-visit accuracy — every accuracy interval in both studies includes zero, and
the plain subtype-only model still has the best accuracy (0.8369) and far and
away the best calibration (0.0065, against 0.0222 for the best banded arm).
Width 0.50's secondary contrast against the unbanded reference tells the same
story as every other width: +0.0177 AUC with an interval excluding zero, but
−0.0057 partial AUC with an interval including zero. **In the usable region of
the ROC curve, no band width — including the optimum — is distinguishable from
not banding at all.**

So the professor's question now has a complete answer. Going below 0.1 is worse.
Going above helps up to about 0.3 and then reverses. The optimum exists, is
located, and is not worth having: it buys a third decimal place of AUC in a
region of the curve nobody operates in, while costing a factor of three in
calibration.

## I. Limitations

- **This is a post-hoc extension.** The pre-specified design was four widths.
  0.50 was added after seeing that the trend had not turned over, and the
  primary four-width analysis stands as its own record.
- **Selection on the evaluation set**, now across five configurations rather
  than four, which slightly worsens the optimistic bias on whichever width wins.
- **The straddling-band confound** from the four-width study is unchanged:
  threshold-on-grid counts across 0.05 → 0.50 are 1, 2, 2, 1, 2, so the
  configurations differ in a second way besides resolution.
- **The optimum is bracketed only coarsely.** The sweep shows a peak at 0.30
  between neighbours at 0.20 and 0.50, but the spacing is wide and the
  differences between 0.20, 0.30 and 0.50 are small. "The optimum is near
  0.3" is supported; "the optimum is 0.30" is not.
- **The turn-over is one point.** 0.50 is a single observation on the falling
  limb. The direction is consistent across seven metrics at once, which is
  reassuring, but a second point above 0.30 would confirm the shape.
- One cohort, one frozen panel, one target, no external validation.

## J. Reproduction

```
python scripts/90_cthmm_ratio_band_width_sensitivity.py \
    --out experiments/<UTC timestamp>_cthmm_ratio_band_width_extended \
    --widths 0.05 0.10 0.20 0.30 0.50 --workers 24
```

1h58m on 24 cores for 5,604 fits; `--resume` reuses the fold checkpoint
and redoes the analysis and figures in about two minutes. Deterministic: seed 42,
2,000 bootstrap resamples, verified byte-identical across consecutive reruns.
Omitting `--widths` reproduces the pre-specified four-width design.

Run on record: `experiments/20261105T025024Z_cthmm_ratio_band_width_extended`.
Code: `src/ppmi_subtype/ratio_bands.py`, `src/ppmi_subtype/next_visit_banded.py`,
`src/ppmi_subtype/ranking_metrics.py`.
Tests: `tests/test_band_width.py` (extended to 0.50), `tests/test_ranking_metrics.py`.
