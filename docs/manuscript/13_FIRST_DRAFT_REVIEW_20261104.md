# First-draft review

**Date** 2026-11-04 · **Reviewed** `manuscript/npj/MANUSCRIPT_FULL_20261104.md`
(992 lines; 7,958 words of prose) · **Target** npj Parkinson's Disease

> **Path correction.** The review request gave the manuscript as
> `docs/manuscript/MANUSCRIPT_FULL_20261104.md`. That file does not exist. The
> assembled manuscript is at **`manuscript/npj/MANUSCRIPT_FULL_20261104.md`**,
> beside the figure and table assets it refers to. `docs/manuscript/` holds the
> stage drafts it is generated *from*.

This is a review of the generated manuscript against its source drafts and
`scripts/81_assemble_manuscript.py`. Assembly succeeding was not taken as
evidence of completeness: the checks below were run against the generated text,
and two of them found content that exists in the drafts but does not reach the
manuscript.

**Headline.** The manuscript **is a complete first draft** and the scientific
argument is sound and answerable from the evidence presented. Four issues
affect scientific validity and should be fixed before the draft goes to
scientific review; the rest is submission preparation. No issue found requires
re-opening a conclusion.

---

## Summary table

| ID | Issue | Class | Priority |
|---|---|---|---|
| B1 | Threshold-perturbation result has no Methods and is not marked post hoc | **validity** | **critical** |
| G1 | ✅ **FIXED** — three contradicting *(supplementary)* labels removed | **validity** | **high** |
| D1 | ✅ **FIXED** — Figure 5a drawn from verified source data | **validity** | **high** |
| B2 | ✅ **FIXED** — the Methods now states it reports no result, and why | validity (minor) | medium |
| C1 | Threshold figures rest on a recomputation, not a saved run | validity | high (deferred) |
| C2 | The 99.87% agreement check is not reproducible | validity | medium (disclosed) |
| H1 | Reference 6 (Erro 2019) was never read beyond Crossref metadata | references | medium |
| H2 | STROBE / TRIPOD+AI deliberately uncited pending E-7 | submission | medium |
| E1 | Tables 2–5 are unformatted source CSVs | presentation | high |
| D2 | Figures 1 and 3 are run figures, not journal-spec | presentation | medium |
| P1 | Working artifacts (assembly note, Asset status) would ship with the paper | presentation | medium |
| F1 | R7's heading asserts more than its hedged body | wording | low |
| G2 | Persistence accuracy rendered 0.7227 and 0.723 | consistency | low |
| F2 | Introduction's roadmap does not foreshadow R5–R8 | structure | low |
| I* | Seven items of author-supplied information | author input | blocking |

---

## A. Sections complete enough for scientific review

All five scientific sections are present and readable end to end. Every
section listed here was checked for the specific failure modes noted, not
merely for presence.

| Section | State | Checked |
|---|---|---|
| **Abstract** | Ready | 147 words against a 150 limit; all eleven quantities traced to saved runs; the two-class 0.84 figure correctly excluded |
| **Introduction** | Ready | 693 words; establishes the clinical problem, the gap and the question; all twelve references first cited here; both negative claims phrased as "we did not find" |
| **Results R1–R8** | Ready, with B1 and D1 | every headline figure re-derived from its source CSV during this session; the A1 orientation correction applied; exploratory analysis labelled in its heading |
| **Discussion** | Ready | 1,479 words including a 469-word limitations subsection covering all twelve Stage 1 items; introduces no new reference; central claim scoped to "in this cohort" |
| **Methods** | Ready, with B1, G1 and B2 | seventeen subsections; every main result except the threshold perturbation has a Methods home |
| **References** | Ready, with H1 | twelve entries, every one with a resolvable DOI, every one cited, numbering 1–12 complete with no gaps |
| **Figure legends / table captions** | Ready | all ten display items have a legend or caption |

**The central question is answered by the evidence presented.** The
Introduction asks whether longitudinal subtype prediction is limited by model
family or by label reliability. The Discussion concludes the latter, and rests
that on three things actually in the Results: the 1.2-point convergence across
five families (R3), the localisation of the entire margin to one stratum (R4),
and the dissociation in which a nine-domain representation matches the hidden
Markov model on the label while beating the label on later outcomes (R5, R7).
The dissociation is what makes the claim more than a restatement of the
convergence, and it is argued explicitly rather than left implicit.

---

## B. Sections incomplete or requiring rewriting

### B1 — The threshold-perturbation result has no Methods *(critical, validity)*

**Location.** Reported at L164–167 (R2) and again at L383–385 (Discussion).

**Problem.** Two things are missing at once. There is **no Methods subsection
describing the procedure** — it is the only main result in the paper without
one, confirmed by mapping every Results subsection onto the Methods — and the
text **nowhere states that the figures are a post-hoc recomputation** rather
than a pre-specified saved analysis. The only use of "recomputation" in the
manuscript (L444) refers to the target label being a recomputation of a
published rule, which is an unrelated statement. The word "post hoc" does not
appear in the manuscript at all.

**Why it matters.** This is the clearest scientific-validity defect in the
draft. A reviewer reading "moving each threshold by approximately 0.02
reclassified 298 of 4,441 labelled visits (6.7%), all of them from the
indeterminate class, which shrank by 67.6%" will look for how the perturbation
was chosen and applied, and find nothing. Worse, the result is load-bearing:
the Discussion's third paragraph uses it to argue that the indeterminate band
is an artefact of arithmetic rather than a clinical state, which is one of the
paper's three main interpretive moves. An undocumented, unregistered
calculation supporting a central argument is the kind of thing that costs
credibility disproportionately to its actual importance.

**Recommended action.** Two steps, only the first of which is in scope now.
(1) Mark the figures post hoc wherever they appear — **done in this review
pass**, see §K. (2) Add a short Methods subsection describing the perturbation
once O-4 is run, stating the pre-specified range and its rationale. Writing
that subsection now is deliberately *not* done: O-4 is deferred, and
describing a procedure that may change when it is actually run would create a
second discrepancy rather than fix one.

**Priority: critical** for the validity fix (done); **high, deferred** for the
Methods subsection.

### B2 — ✅ FIXED — growth-model Methods described a model with no result *(medium)*

**Location.** L677, `### Hierarchical Bayesian growth model *(supplementary)*`,
roughly 15 lines of Methods.

**Problem.** No growth-model result appears anywhere — not in the main Results,
not in a supplementary section, because no supplementary section exists yet. A
reader is given the specification, the sampler, the identifiability argument
about observation and process error, and then never sees an estimate.

**Why it matters.** Less than B1, because nothing is *claimed* on the back of
it, so no conclusion is at risk. But it reads as an orphan: either the model
earned a place in the paper, in which case its result and its single-chain
diagnostic limitation both belong somewhere visible, or it did not, in which
case the Methods text is dead weight that invites the question "where are the
numbers?".

**Resolved.** The Methods subsection now states plainly that **no result from
this model is reported anywhere in this manuscript**, that it is described so
the modelling actually performed is on the record, and that it is excluded
because it ran as a single chain with no R̂ and no effective sample size —
"not an adequate basis for any inferential claim". The model was neither cut
nor refitted, so the deferred decision about it stays open; what changed is
that the orphan is now explained rather than unexplained. This also puts the
C3 single-chain limitation into the manuscript text for the first time.

**Priority: medium — closed.**

### B3 — Introduction's roadmap does not foreshadow R5–R8 *(low)*

**Location.** L96–101, the roadmap sentence.

**Problem.** The roadmap promises the reversion measurement, the
threshold-sensitivity measurement and the nine-specification comparison. It
does not mention the two-class restriction (R5), the history and retraining
analyses (R6), the continuous representation (R7) or the per-participant
analysis (R8) — four of eight Results subsections, including R7, which supplies
the dissociation the Discussion leans on hardest.

**Why it matters.** Structure only. A reader arrives at R7 unprepared for it,
and the Discussion's strongest argument rests on a result the Introduction
never signalled.

**Recommended action.** One clause added to the roadmap. Low cost, and worth
doing before the draft circulates.

**Priority: low.**

---

## C. Numerical claims needing verification against source outputs

Most of the manuscript does **not** belong in this section, and that is worth
stating plainly: every headline quantity in R1–R8 was re-derived from its
source CSV during this session, cell by cell, and the Stage 12 audit records
the comparison. Specifically verified exact: all nine transition probabilities
with patient-clustered intervals and design effects; pooled reversion and all
six change types; the margin decomposition across three models and three
strata; the harmonised metrics for all fourteen specifications; representation
dimensionality; the Hoehn & Yahr and MCI outcome deltas; and every cohort
count. Two claims remain outstanding.

### C1 — Threshold figures rest on a recomputation *(high, deferred)*

The figures in B1 above (298/4,441, 6.7%, 67.6%, 97.74%) were computed from
the frozen panel but never saved as a run, so they have no manifest, no seed
record and no checksum. **Action:** run O-4 — deferred by instruction. Until
then they carry the post-hoc marking added in §K.

### C2 — The 99.87% three-class agreement is not reproducible *(medium)*

**Location.** L120–122 (R1).

`data/validation/` is empty, so the archived 2020 release the check was made
against is gone. This is **already disclosed** in the Discussion's limitations
("one internal agreement figure quoted in the cohort section cannot currently
be recomputed"), which is the honest minimum. **Action:** restore the archived
release and rerun, or leave the disclosure. Note that R1's editorial framing
already instructs that this figure must not be promoted to "validation", and
the manuscript respects that.

---

## D. Figures missing or needing redrawing

### D1 — ✅ FIXED — Figure 5a had never been drawn *(high, validity)*

**Location.** Referenced at L298; Asset status at L985 records
`NOT DRAWN — source data only`.

**Problem.** R7 cites Fig. 5a for the claim that seven principal components are
needed to span 90% of across-participant variance, with PC1 at 42.2% of the
levels and 37.2% of the rates. The underlying numbers are verified — they come
from `representation_dimensionality.csv` and reproduce exactly across two
independent runs — but **no figure exists**. An earlier candidate,
`manuscript/figures/fig09_representation.png`, was checked and is a different
quantity (per-domain rate statistics, not component variance), so it cannot be
substituted.

**Why it matters.** It is classed as a validity issue rather than presentation
because the claim "it is not a severity score in disguise" is the premise for
treating the representation as genuinely multidimensional, and the Discussion
uses that premise. A cited figure that does not exist cannot support it.

**Resolved.** Drawn by a new `figure_5a()` in
`scripts/80_manuscript_assets.py`, from
`representation_dimensionality.csv` in the definitive rerun. It shows
per-component variance share for level and rate coordinates as grouped bars,
cumulative curves on a second axis, and the 90% crossing at seven components.
Both values R7 quotes are annotated on the figure itself, so the text and the
picture cannot drift apart. Four new checks were added and pass: PC1 level
share matches 42.2%, PC1 rate share matches 37.2%, seven components reach 90%
for both, and the shares sum to one — 16 of 16 checks pass overall. The asset
is `manuscript/npj/fig5a_representation_dimensionality.png`, with its tidy data
beside it, and the Asset status table now lists it submission-ready.

Two layout defects were caught by inspecting the rendered figure rather than
trusting that it had been generated: the 90% annotation collided with the
threshold line and both cumulative curves, and the legend was missing its
cumulative entry because the proxy artist was added after `legend()` was
called. Both fixed.

**Priority: high — closed.**

### D2 — Figures 1 and 3 are run figures *(medium, presentation)*

Both exist and both show the right content, but they were drawn by their
analysis scripts rather than to journal specification. Figures 2, 4 and 5b are
submission-ready. **Action:** redraw 1 and 3 in the same pass as D1.

---

## E. Tables needing formatting or verification

### E1 — Tables 2–5 exist only as source CSVs *(high, presentation)*

| Table | Source | State |
|---|---|---|
| 1 | `manuscript/npj/table1_cohort_characteristics.csv` | submission-ready |
| 2 | E-2 transition bootstrap run | CSV only |
| 3 | E-8 `harmonised_metrics.csv` | CSV only |
| 4 | `outputs/joint_subscore/` + AUC metrics | CSVs only |
| 5 | patient-level audit composition CSV | CSV only |

**Verification status is good** — all four have captions written, all four are
referenced in the text, and their values were checked against source during
this session. What is missing is formatting: titles, consistent column labels,
decimal precision, units, sample sizes and interval notation. **Action:**
format in one pass; do not alter any value. **Priority: high** for submission,
**not** a validity issue.

---

## F. Claims possibly stronger than the evidence

The draft is, on the whole, carefully hedged — the central claim is scoped to
"in this cohort", rate-of-change results are reported as not established,
outcome differences are stated as upper bounds, and the five-state model's
convergence failure is now disclosed. Two wording issues remain.

### F1 — R7's heading asserts more than its body *(low)*

**Location.** L295, `### A continuous representation carries information the
label does not`.

The body is properly hedged: the coordinate was selected on the same data it is
compared against, so each difference is an upper bound, and the rate
coordinates were not established as informative on any outcome. The heading
states the conclusion flatly. Headings are read and remembered
disproportionately. **Action:** soften to something like "A continuous
representation discriminates later outcomes better than the label", which is
what was measured. **Priority: low.**

### F2 — The title states the conclusion *(low, author's call)*

"Label reliability rather than model capacity limits longitudinal prediction…"
asserts the finding. The body supports it for this cohort and this rule, and
the Discussion scopes it correctly. Some reviewers read a conclusion-stating
title as overreach regardless. Two neutral alternatives are recorded in
`11_ABSTRACT_TITLE_DRAFT_20261104.md`. **This is an author decision, not a
defect.**

---

## G. Contradictions between sections

### G1 — ✅ FIXED — four *(supplementary)* labels contradicted the Results *(high, validity)*

**Location.** Methods L677, L692, L711, L745.

**Problem.** Four Methods subsections carry the marker *(supplementary)*. For
three of them, the Results present those models as **primary evidence**:

| Methods subsection | Labelled | Where its results actually appear |
|---|---|---|
| Subscore models | *(supplementary)* | **R3** — two subscore GPs and the joint subscore LMM are 3 of the 9 specifications in the primary comparison, and the GP is the top scorer at 0.7575 |
| Continuous multi-domain clinical representation | *(supplementary)* | **R5 and R7** — R7 is an entire main Results subsection |
| History-window and retraining analyses | *(supplementary)* | **R6** — an entire main Results subsection |
| Hierarchical Bayesian growth model | *(supplementary)* | nowhere (see B2) |

**Why it matters.** This is a validity issue, not a typographical one. The
Methods tell a reviewer that three of the five model families are
supplementary, while the Results and Discussion rest the paper's central claim
on exactly those families. Someone assessing whether the convergence finding is
adequately supported would be actively misled about the evidential status of
most of its inputs. It also reads as though the main-versus-supplementary
allocation was never settled — which, per the current deferral list, it has not
been.

**Resolved, by the minimum correction.** The three labels that contradicted
the Results were removed; the growth model keeps its label, because for that
model it is accurate. Methods and Results now agree on the evidential status of
every model family.

This deliberately does **not** make the deferred allocation decision. Nothing
was promoted to supplementary or demoted from it, and no supplementary section
was created. All that changed is that the Methods stopped describing three
families as supplementary while the Results presented them as primary. If the
later allocation review moves any of them, the labels should be reinstated
there — correctly, and alongside an actual supplementary section.

**Priority: high — closed.**

### G2 — Persistence accuracy rendered two ways *(low)*

0.7227 in Results, 0.723 in the Discussion. Same verified value, two
precisions. **Action:** harmonise. **Priority: low.**

---

## H. References and citations requiring attention

### H1 — Reference 6 (Erro 2019) was never read *(medium)*

Only Crossref metadata was verified; the abstract was never retrieved. It is
cited in the Introduction for the general claim that subtyping schemes are
unstable over time — a claim that references 4, 5 and 7 already carry, each of
which was verified from its abstract or full text. **Action:** obtain the
abstract, or drop the citation. Dropping it costs the argument nothing.
**Priority: medium** — citing a source one has not read is a real integrity
issue even when the claim is well supported elsewhere.

### H2 — STROBE and TRIPOD+AI are deliberately uncited *(medium, submission)*

Both were verified at Stage 3 and both are **intentionally absent**, because
the manuscript has not been assessed against either checklist (E-7 never run).
Citing a reporting standard one has not been checked against would assert a
compliance that does not exist. This is the correct call, but it leaves a
visible gap: many journals request the completed checklist at submission.
**Action:** run E-7 — deferred — then add both references. The Methods'
Reporting section should name the standards it followed only after that.

### Citation mechanics — clean

All twelve references carry a resolvable DOI, all twelve are cited, numbering
runs 1–12 with no gaps or out-of-range markers, and order of first appearance
holds (all twelve are first cited in the Introduction). The mathematical
superscripts in Methods and Results — 10⁻¹², χ², σ²_L·Δ³ — were checked and are
not corrupted into citation markers; only two markers are inserted by name
anchor, Simuni⁴ in Results and Jackson¹¹ in Methods.

---

## I. Author-provided information still missing

Seven items, listed in full with their verification status in
`14_AUTHOR_INFORMATION_CHECKLIST_20261104.md`. In brief: ethics, funding,
competing interests, CRediT contributions, PPMI acknowledgement wording,
repository hosting, and the author list's order and affiliations. **All seven
are blocking for submission and none can be resolved from repository
materials.** None is a scientific defect.

### P1 — Working artifacts would ship with the paper *(medium, presentation)*

Two blocks in the generated file are internal and must not reach a journal: the
assembly note at L3 ("Assembled by `scripts/81_assemble_manuscript.py`…") and
the entire **Asset status** section at L975–989, which lists local experiment
paths and the state of each display item. Both are useful now and wrong later.
**Action:** add a `--submission` mode to the assembly script that omits them,
rather than deleting them by hand from generated output.

---

## J. Analyses to remain deferred

Per the current decision, and recorded here so the deferral is explicit rather
than forgotten:

| Deferred | Consequence while deferred |
|---|---|
| O-4 threshold-sensitivity recomputation | C1 stands; B1's Methods subsection cannot be written; the figures carry a post-hoc marking |
| E-7 STROBE / TRIPOD+AI checklists | H2 stands; no reporting standard is cited |
| Hierarchical growth model investigation | B2 stands; the Methods subsection remains an orphan |
| Five-state HMM exclusion decision | the disclosure added on 2026-11-04 stays as written; **no change made** |
| Main vs supplementary allocation | G1 is flagged, not corrected; the three contradicting labels remain |

None of these deferrals affects a conclusion. Each leaves a documented gap
rather than an unknown.

---

## K. Changes made

Four changes, each closing a validity issue identified above. Everything else
is reported, not changed.

**1. Threshold figures marked post hoc (B1, partial).** They now read "In a
post-hoc calculation on the frozen panel…" in R2 and "in a post-hoc
calculation" in the Discussion, and R2 adds that the perturbation was not
pre-specified and the figures are exploratory. The Methods subsection B1 also
calls for is **not** written, because O-4 is deferred and describing a
procedure that may change when it is run would create a second discrepancy.

**2. Figure 5a drawn (D1, closed).** New `figure_5a()` in
`scripts/80_manuscript_assets.py` with four new checks; 16 of 16 pass.

**3. Three *(supplementary)* labels removed (G1, closed).** The minimum
correction, which makes Methods agree with Results without making the deferred
allocation decision.

**4. Growth-model Methods explained (B2, closed).** It now states that no
result from the model appears anywhere and why — single chain, no R̂, no
effective sample size. The model was neither cut nor refitted.

### Not changed, by design

The five-state HMM disclosure (left exactly as documented); every exploratory
result, R8 retained with its label; the scientific thesis; the Methods
subsection for the threshold analysis (B1, with O-4); R7's heading (F1); the
0.7227 / 0.723 rendering (G2); the Introduction roadmap (B3); the working
artifacts (P1); Tables 2–5 (E1); Figures 1 and 3 (D2); reference 6 (H1).

### Files touched

| File | Change |
|---|---|
| `08_RESULTS_DRAFT_20261104.md` | post-hoc marking (R2) |
| `10_DISCUSSION_DRAFT_20261104.md` | post-hoc marking |
| `07_METHODS_DRAFT_20261104.md` | three labels removed; growth-model disclosure |
| `scripts/80_manuscript_assets.py` | `figure_5a()` added |
| `scripts/81_assemble_manuscript.py` | Asset status row for Fig. 5a |
| `manuscript/npj/MANUSCRIPT_FULL_20261104.md` | regenerated |
| `manuscript/npj/fig5a_*.png`, `.csv` | new assets |

**Nothing was committed and nothing was pushed.**

### What remains open

Validity: **B1's Methods subsection** (needs O-4) and **C1** (same). Everything
else outstanding is presentation (E1 tables, D2 figures, P1 artifacts),
wording (F1, G2, B3), references (H1, H2 — H2 needs E-7), or author input
(§I, seven blocking items).
