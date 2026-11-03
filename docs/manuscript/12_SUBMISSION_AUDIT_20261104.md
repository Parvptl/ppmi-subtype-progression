# Stage 12 — Submission audit

**Date** 2026-11-04 · **Audited** `manuscript/npj/MANUSCRIPT_FULL_20261104.md`
(974 lines, 7,787 words of prose) · **Target** npj Parkinson's Disease

This audit re-derived the manuscript's headline quantities from their source
runs rather than from the stage drafts, so that an error copied consistently
through four drafts would still be caught. **One substantive finding emerged
that changed a reported number** (A1, since fixed and confirmed by a full
rerun). A second finding (A2) turned out to be an artifact of my own
comparison and is retracted; tracing it exposed a real reproducibility trap in
the pipeline (A2′). Everything else is asset work, submission paperwork, or
clarity.

---

## 1. The clustering exclusion — verified clean

The brief's absolute exclusion was checked directly against the assembled
manuscript. A search for `cluster`, `k-means`, `kmeans`, `silhouette`,
`co-cluster`, `mixture`, `GMM`, `unsupervised` and `subgroup discovery`
returned **seven hits, none of them from the clustering investigation**:

| Hits | What they are |
|---|---|
| 4 | "cluster bootstrap" / "patient-clustered bootstrap" — the resampling method used for every interval |
| 1 | "clustered data" — within-participant dependence, in the limitations |
| 1 | "an even mixture of tremor-dominant and PIGD outcomes" — ordinary English, describing label composition in R8 |
| 1 | Figure/table captions repeating the bootstrap method |

No experiment directory from the clustering line is referenced anywhere in the
manuscript or its Asset status table. Dadu 2022 is cited in the Introduction
for *published external* work on data-driven subtypes; that is literature
context verified at Stage 3, not this project's excluded investigation.

**Verdict: the exclusion holds.**

---

## 2. Numerical verification against source runs

Re-derived from the saved runs, not from the drafts. Every row below was
checked cell by cell.

| Quantity | Source | Result |
|---|---|---|
| All nine transition probabilities and intervals | `e2_.../transition_probabilities.csv` | ✅ exact — TD→TD 0.808 [0.788, 0.826]; Ind→Ind 0.192 [0.148, 0.234]; PIGD→PIGD 0.702 [0.661, 0.738] |
| Design effects (largest, smallest) | same | ✅ exact — PIGD→PIGD 1.463, TD→TD 1.383, only sub-1 cell Ind→PIGD 0.909 |
| Pooled reversion and all six change types | `e2_.../other_quantities.csv` | ✅ exact — 0.3522 [0.3180, 0.3880]; TD→Ind 0.560, PIGD→Ind 0.500, TD→PIGD 0.320, PIGD→TD 0.393 |
| Margin decomposition, three models × three strata | `e8_.../margin_decomposition.csv` | ✅ exact — CT-HMM 94.3% share, 6/99/0 gained, 0.984/0.003/0.969 identical; chain 100%/83 cases; GP 9 and 8 |
| Nine specifications, five families, spread | `e8_.../harmonised_metrics.csv` | ✅ exact — 14 rows, 5 baselines, 9 fitted; 0.7453–0.7575; persistence 0.7227; macro-F1 0.5559 highest |
| Representation dimensionality | `clinical_profile_cv/representation_dimensionality.csv` | ✅ exact — 7 components for 90%, PC1 0.4224 level / 0.3723 rate |
| H&Y ≥ 3 outcome difference | `clinical_profile_cv/outcome_associations.csv` | ✅ exact — 0.6845 − 0.5794 = +0.1051 |
| MCI outcome difference | same | ✅ exact — 0.7098 − 0.5494 = +0.1604 |
| Motor-complications difference | same | ⚠️ **see A1** — manuscript +0.0347, consistent convention gives +0.0272 |
| Cohort counts | `outputs/panel_summary.txt` | ✅ exact — 1,080 / 934 / 4,441 / 3,361; PIGD 25.0% of labelled visits, 18.1% at baseline |
| Undefined ratio, label loss | `panel_summary.txt`, Methods M4 | ✅ exact — 725 of 4,441 = 16.3%; 1,195 of 1,333 = 89.6% |

---

## 3. Findings, ranked

### A — Changes a reported number

**A1. ✅ RESOLVED 2026-11-04.** Figure 5b and Table 5 handled AUC orientation
inconsistently, and one reported difference was inflated as a result.

The comparator coordinates are selected and reported on an
**orientation-corrected** basis (`auc_abs` = max(AUC, 1 − AUC), which is what
makes a coordinate like `level_moca` usable at 0.7098 when its raw AUC is
0.2902). The subtype-label reference is reported **raw**.

For three of four outcomes this is invisible, because the label's raw AUC
exceeds 0.5 and the two conventions coincide. For **motor complications it does
not**: the label's raw AUC is 0.4963 and its orientation-corrected value is
0.5037.

| Convention for the reference | Difference reported |
|---|---|
| Raw (what Fig. 5b and the manuscript use) | **+0.0347** [−0.0165, +0.0830] |
| Orientation-corrected, matching the coordinates | **+0.0272** |

**The conclusion is unaffected** — the interval includes zero on either
convention, and R7 already reports motor complications as the outcome where the
difference is not established. But the quoted point estimate rests on applying
a favourable orientation to the comparator and not to the reference, which is
the direction that flatters the representation.

**What was done.** The orientation-corrected reference was adopted, as the
conservative and internally consistent choice given that the comparator's
selection is already disclosed as an upper bound.
`src/ppmi_subtype/clinical_profile.py::best_component_vs_subtype` now applies
the same one-time flip to the reference that it already applied to the
comparator — fixed before the bootstrap, so orientation is a property of the
score rather than something re-chosen per replicate. The function has exactly
one call site, so the blast radius is these eight contrasts.

`scripts/82_outcome_contrast_orientation_fix.py` recomputed all eight,
**running both arms under one version of the library** so that the change is
isolated from any unrelated library drift. Results in
`experiments/20261103T212122Z_a1_outcome_contrast_orientation/`:

| Contrast | Before | After |
|---|---|---|
| motor complications, level | +0.0347 | **+0.0272** [−0.0394, +0.0941] |
| motor complications, rate | +0.0408 | **+0.0334** [−0.0254, +0.0903] |
| all six other contrasts | — | **shift exactly 0.0000** |

Nine checks pass, including that only motor complications moved, that its
interval still includes zero, that every rate interval still includes zero and
that three of four level intervals still exclude it. Figure 5b was rebuilt;
`fig2` and `fig4` are byte-identical, confirming the rebuild was surgical. R7
now reports +0.027 and the rate list reads +0.053, +0.043, +0.033 and +0.036,
and notes that the label discriminates motor complications at chance (0.504).

**No scientific conclusion changed.** The difference moved in the conservative
direction on the one outcome the manuscript already reported as not
established.

**A2. ❌ RETRACTED 2026-11-04 — the discrepancy was an artifact of my own
comparison, not of the library.** Replaced by A2′ below.

*What I claimed:* that the October CV run's saved rate contrasts were not
reproducible under current code, with `mci`/`rate` drifting −0.0020, and that
the cause was that the run executed uncommitted code.

*What is actually true:* a full rerun of `scripts/64_clinical_profile_cv.py` at
HEAD (`experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/`, 352 s, same
folds, repeats and workers) reproduces the October run **exactly — max
\|delta\| = 0 across all eighteen CSVs** — with one exception,
`h3_component_vs_subtype.csv`, whose only moved quantities are the
motor-complications reference AUC, difference and interval. That is exactly and
only the A1 fix. `outcome_associations.csv` is byte-identical, so `auc` did not
change and there is no library drift.

The drift I measured came from `scripts/82_outcome_contrast_orientation_fix.py`
reading the run's **saved** `outcome_table.csv`. See A2′.

*The provenance observation stands, but is now moot.* The October run really
did record commit `3a860ca` (2026-10-16), at which
`src/ppmi_subtype/clinical_profile.py` did not exist — it was first committed
in `19bf8aa` five days later — so it ran uncommitted code. That was a genuine
provenance gap. The rerun closes it empirically: the numbers are confirmed
identical and now carry a commit that contains the module. The provenance audit
of the other nine cited runs (table retained below) found no other instance.

| Section | Run commit | Date | Modules present |
|---|---|---|---|
| R1, R2 | `ca3497ec` | 2026-11-03 | ✅ |
| R3, R4 | `82dddadc` | 2026-11-04 | ✅ |
| R5 (ROC) | `3775a395` | 2026-10-21 | ✅ |
| R5, R7 (profile CV) | `3a860ca8` | 2026-10-16 | ❌ `clinical_profile.py` — superseded by the rerun |
| R6 ×4 | `3775a395` | 2026-10-21 | ✅ |
| R8 | `3775a395` | 2026-10-21 | ✅ |

---

**A2′. `scripts/64_clinical_profile_cv.py` saves a rounded table while
computing on full precision, and reading it back changes results.**

Line 547 passes the in-memory `outcome_table` to `outcome_associations` and
`best_component_vs_subtype`, but writes the file as
`outcome_table.round(4).to_csv(...)`. Four decimal places is harmless for level
coordinates, whose values are of order 1, and **destructive for rate
coordinates, whose values are of order 10⁻³**: rounding collapses 83 distinct
`rate_moca` values to exactly zero, which perturbs the rank statistic the AUC
is built from.

That is the whole explanation for "only rate contrasts moved", and the shift
tracks magnitude, not tie structure as I first supposed:

| Selected component | Typical magnitude | Shift from reading the rounded file |
|---|---|---:|
| `rate_moca` | ~3 × 10⁻³ | **−0.0020** |
| `rate_updrs1` | ~8 × 10⁻² | +0.0001 |
| `rate_updrs2` | ~2 × 10⁻¹ | +0.0001 |
| all four level coordinates | ~1 | 0.0000 |

**It cost a wrong number in a draft.** R7 briefly read "+0.043" for the `mci`
rate difference, computed from the rounded file; the correct value is **+0.045**
at full precision, which is what both the October run and the rerun give and
what the draft said before I touched it. R7 now reads +0.053, +0.045, +0.033
and +0.036, sourced from the rerun. The motor-complications *level* figure
(+0.0272) is identical at both precisions, so A1's substantive correction is
unaffected; its *rate* figure moved by 0.0001.

**✅ FIXED 2026-11-04.** `scripts/64_clinical_profile_cv.py` now saves
`outcome_table.csv` at full precision, matching the convention it already used
for its three other per-patient tables (`representation.csv`,
`profile_predictions.csv`, `derived_label_predictions.csv`), which were never
rounded. One line.

**The trap was worse than the drift suggested.** Rounding did not merely
perturb `rate_moca`; it annihilated whole columns, because rate coordinates
span four orders of magnitude:

| Rate column | Exact zeros when rounded | Exact zeros at full precision |
|---|---:|---:|
| `rate_tremor` | **856 of 934** | 0 |
| `rate_ess` | 182 | 0 |
| `rate_gds` | 125 | 6 |
| `rate_scopa` | 95 | 9 |
| `rate_moca` | 83 | 3 |
| `rate_pigd` | 29 | 0 |

The smallest non-zero rate magnitude in the saved table rises from 1 × 10⁻⁴ to
5.4 × 10⁻⁹. No reported number used `rate_tremor`, so nothing in the manuscript
was affected by that column — but anyone reusing it from the saved file would
have been working with a 92%-zero vector.

**Verified both ways** with `scripts/84_verify_outcome_table_roundtrip.py`,
which recomputes the eight contrasts from the *saved* table and compares them
against the run's own saved results:

* on the rounded table: **FAIL** — four contrasts differ, every one of them a
  `rate` contrast, worst `mci`/`rate` at 0.0023, all four `level` contrasts
  exact;
* on the full-precision table (`experiments/20261103T220147Z_a2p_outcome_table_fullprec/`):
  **PASS** — all forty comparisons exactly zero.

**The fix is computationally inert**, as it must be, since the analysis always
used the in-memory table: a full rerun with the fix reproduces the previous
rerun on every result file, `h3_component_vs_subtype.csv` included. Only
`outcome_table.csv` changes, in 18,486 of 23,571 numeric cells, by at most
5 × 10⁻⁵ — the 4dp bound.

**Scope of the trap, bounded.** Sixteen scripts write rounded CSVs, but the
rounded files are results tables read for display or comparison, where four
decimals suffice. Every file that is read back and *recomputed from* — the
prediction files `next_visit_predictions.csv`,
`derived_label_predictions.csv`, `profile_predictions.csv`,
`clinical_state_predictions.csv`, `window_predictions.csv` and
`condition_predictions.csv`, which scripts 66–75 and 79 all consume — is
written with a plain `to_csv` at full precision, confirmed in the saved data.
`outcome_table.csv` was the only data table that was lossy, and it is the only
one this affected.

*Also:* `scripts/82_...py` is superseded and marked so in its own docstring;
Figure 5b and R7 are now built from the definitive run above.

### B — Blocks submission, not science

**B1. All five back-matter sections are absent (C5).** Ethics declarations,
funding, competing interests, author contributions (CRediT) and
acknowledgements. Each is explicitly marked ABSENT in the assembled manuscript
rather than omitted, and the audit confirms **nothing is falsely asserted**: no
ethics approval, consent, funding, trial registration or external validation
appears anywhere. PPMI also requires specific acknowledgement wording that is
not in the repository. **Authors must supply all five.**

**B2. ❌ RETRACTED 2026-11-04 — the Gaussian-process modules were never
untracked.** This entry claimed they were untracked and never pushed. All four
(`gpr.py`, `gpr_loo.py`, `gpr_plots.py`, `trajectory.py`), scripts 11–16 and
three test files were added on **2026-09-30 in `dada934`** (21 files, 6,224
insertions, alongside eight `docs/gpr_audit/` documents), and that commit is
**already on `origin/main`**. `git status --short` is empty and the repository
contains no untracked files whatsoever; nothing GP-related is in `.gitignore`.

R3's subscore GP depends on `scripts/18_subscore_multioutput_gp.py` and the
tracked modules `mogp.py`, `gpr.py`, `trajectory.py` and `subtype.py`. The
harmonised comparison itself re-scores the saved predictions and refits no GP,
so the dependency is on code that exists in the published history.

**Why this matters beyond the one claim.** The error originated in the Stage 1
audit, which asserted it without running `git ls-files`, and this audit
repeated it — an audit whose purpose was to catch exactly this kind of thing.
Every numeric claim in the manuscript was re-derived from its run, but
repository claims were taken on trust. That asymmetry is the lesson: a
statement about the repository needs a command, not a recollection.

**What is actually open** is narrower and is not a code problem: the repository
URL, and whether it will be made public or archived with a DOI at acceptance.
Recorded as an author query.

**B3. The author list is unconfirmed.** Taken from the CARE AI 2026 abstract
and marked as such.

### C — Asset work

| | Item | State |
|---|---|---|
| C1 | **Fig. 5a** | **Not drawn.** Source data verified (`representation_dimensionality.csv`); no figure exists. `manuscript/figures/fig09_representation.png` was checked and is a *different* quantity (per-domain rate statistics), not PC variance. |
| C2 | Fig. 1, Fig. 3 | Exist as run figures; need redrawing to journal specification. |
| C3 | Tables 2–5 | Source CSVs only; not formatted as manuscript tables. |

Submission-ready today: Fig. 2, Fig. 4, Fig. 5b (pending A1), Table 1.

### D — Evidential gaps

**D1. O-4 — the threshold-sensitivity figures are a recomputation, not a saved
run.** R2's claim that moving each threshold by ≈0.02 reclassifies 298 of 4,441
visits (6.7%), all from the indeterminate class, shrinking it by 67.6%, is
currently a post-hoc calculation from the frozen panel. Run it as a saved
experiment or mark it post hoc in the caption. It is load-bearing for the
Discussion's third paragraph.

**D2. E-7 was never run, so no reporting checklist is claimed.** STROBE and
TRIPOD+AI were both verified at Stage 3 but are **deliberately not cited**,
because citing a standard the manuscript has not been assessed against asserts
a compliance that does not exist. Run E-7, then add both references. Many
journals request the completed checklist at submission.

**D3. C6 — the 99.87% three-class agreement check is not reproducible**
(`data/validation/` is empty). This is **disclosed in the manuscript's
limitations**. If the archived release is restored, rerun the check and delete
that sentence.

**D4. Erro 2019's abstract was never retrieved** — only Crossref metadata was
verified. It supports a general instability claim that references 4, 5 and 7
already carry. Obtain the abstract or drop the citation; dropping it costs the
argument nothing.

**D5. C3 — the growth model has no publishable MCMC diagnostics** (one chain,
no R̂ or ESS). **No growth-model result is quoted anywhere in the manuscript**,
so there is no current exposure. If it appears in supplementary material it
must be labelled descriptive and the gap disclosed.

**D6. ✅ FIXED 2026-11-04 — and this audit had missed it.** The five-state
hidden Markov model's accuracy of 0.7513 was quoted in R3 beside the
three-state model's 0.7539 with no caveat, and Methods said only that it "was
fitted identically as a sensitivity analysis". Measured from `train_converged`
in `outputs/loo_5hidden/loo_predictions_all_models.csv`, that model **reported
convergence in 404 of 934 folds (43.3%)**, against **934 of 934** for the
three-state model and for every chain and reference rule in the same run, and
**1,990 of its 3,361 predictions (59.2%) come from folds that did not
converge** — they were retained, not discarded. A reader would have taken the
two scores as comparable.

Both Methods and R3 now disclose it. R3 also states what a reviewer will ask
next: the five-state accuracy is **interior** to the range, sixth of the nine,
so dropping it leaves the 1.22-point spread unchanged and the convergence
finding does not rest on it.

Two things worth recording about how this was found. First, **this audit did
not flag it** — §2 verified that 0.7513 matched its source, which it does, but
never asked whether the figure was comparable. Checking a number against its
run is not the same as checking that it means what the sentence implies.
Second, the 404/934 figure also appears in `docs/cluster_phase/`, where it is
marked "Superseded: budget-limited, multimodal". That folder is excluded from
the manuscript's evidence, so the figure was **re-measured from the current
run** rather than taken from there, and the current run independently gives the
same 404/934 with the additional 59.2% share.

### E — Clarity

**E1. "9.9%" denotes two different quantities.** 334 transitions whose
*current* label is indeterminate (9.937%) and 332 whose *next* label is
indeterminate (9.878%) both round to 9.9%. The Discussion says "current"
explicitly; the Abstract's "transitions recorded indeterminate" does not.
Tighten the Abstract to "recorded indeterminate at the prediction point" if the
word budget allows — it currently has three words of margin.

**E2. Persistence accuracy appears as 0.7227 (Results) and 0.723
(Discussion).** Rounding variants of one verified value; harmonise the
precision.

**E3. Two R7 values sit at exact three-decimal ties.** The H&Y level AUC is
stored as 0.6845 and the cognitive-diagnosis rate difference as 0.0355, so the
third decimal depends on the rounding convention — the manuscript uses
round-half-up (0.685, +0.036), which is what the draft has always used.
The ties are in the 4dp results table, not the data; now that A2′ is fixed the
underlying table is full precision, so the ties can be broken properly if
anyone asks. Harmless as reported — if a reviewer queries it, the answer is the
convention.

---

## 4. Compliance checks

| Check | Result |
|---|---|
| All 12 references carry a resolvable DOI | ✅ 12/12 |
| All 12 references actually cited; numbering complete | ✅ markers 1–12 all used, no gaps |
| Citation order = order of first appearance | ✅ all 12 first cited in the Introduction |
| Mathematical superscripts not corrupted into citations | ✅ 10⁻¹², χ², σ²_L·Δ³ intact; only two name-anchored markers inserted |
| Every figure and table referenced in text | ✅ Fig. 1–5, Tables 1–5 |
| Every figure and table has a legend or caption | ✅ 10/10 |
| No forbidden novelty or hype phrasing | ✅ none |
| No causal language on observational findings | ✅ none |
| No unqualified "significant" | ✅ none |
| No fabricated ethics, consent, funding, registration or external validation | ✅ none asserted |
| Absent required sections explicitly marked | ✅ 5/5 marked ABSENT |
| No editorial or stage-process text leaked into the manuscript | ✅ clean |
| Title ≤ 15 words, no punctuation | ✅ 14 words, punctuation-free |
| Abstract ≤ 150 words | ✅ 147 |
| Unperformed analyses absent | ✅ the three identified at Stage 7 remain absent |

**Journal limits remain unverified.** The 150-word and 15-word rules came from
a search result; the live Nature page was behind an authentication redirect.
Figure, table and reference limits, and the current AI-disclosure policy, were
never verified. **Re-check all of these at submission.**

---

## 5. State

**Submission-ready:** the manuscript text. All five sections are drafted,
internally consistent, and every headline quantity traces to a saved run.

**Not submission-ready:** the paperwork (B1, B3) and four of ten display items
(C1–C3). A1 is fixed and verified against a full rerun; A2 is retracted; A2′ is
fixed and verified by round-trip in both directions; **B2 is retracted — the
code was always tracked**, leaving only the repository's hosting and archival
as an author query.

**Order I would work in:** B1 and B3 next, which need the authors and are not
analysis work. Then C1–C3 as a single asset-building pass. D1 and D2 are
experiments worth running before a reviewer asks. D3–D5 and E1–E3 are small
and can ride along.

**Nothing in this audit requires re-opening a scientific conclusion.** A1
changes a point estimate on the one outcome the manuscript already reports as
not established, in the conservative direction.

**Two of this audit's own findings were wrong** (A2 and B2), and both were
wrong the same way: a claim about the repository or its artifacts that was
carried forward rather than re-derived. A third issue (D6) it missed entirely,
because it checked whether each number matched its run and not whether the
sentence around the number was fair. The numeric claims, which were all
re-run from source, held up. Any future audit here should treat "the repository
contains / does not contain X" as a claim needing a command, exactly as it
treats a confidence interval.
