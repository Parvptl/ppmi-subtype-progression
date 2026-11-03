# Stage 8 — Introduction, drafted

**Date** 2026-11-04 · **Target** npj Parkinson's Disease, Article · **Status**
first full draft

Written after the question and the results were fixed, as the brief requires.
Every claim is tied to a reference verified at Stage 3; §A maps each sentence
to its source and §B records the claims deliberately *not* made.

Conventions as before: **manuscript text is the numbered section**; editorial
notes are indented blockquotes and are not manuscript text.

---

## Introduction

Parkinson's disease is clinically heterogeneous, and the most widely used
summary of that heterogeneity is the motor subtype: patients are classified as
tremor dominant, postural-instability/gait-difficulty (PIGD), or indeterminate.
The classification is arithmetically simple. Eleven MDS-UPDRS items are
averaged into a tremor score, five into a PIGD score, and the ratio of the two
is compared against a pair of thresholds, with the interval between them
forming the indeterminate band¹. Its appeal is that it is cheap, interpretable
and computable from an examination that is performed anyway.

That label is used well beyond description. It stratifies cohorts, defines
subgroups for biomarker comparison, and is treated as a longitudinal quantity
whose change marks progression — a role that a recent international appraisal
of Parkinson's disease subtype classifications examined across many such
systems, and for which it found the evidence base thin². Whether the PIGD
construct identifies a distinct phenotype at all has been separately
questioned³.

Several groups have shown the label to be unstable over follow-up. In the de
novo PPMI cohort, 39% of patients classified PIGD and 18% classified tremor
dominant changed subtype within twelve months⁴. Over four years in the same
cohort, the tremor-dominant proportion fell from 68% to 45% while PIGD rose
from 23% to 44%⁵. Comparable instability has been reported for PIGD and
akinetic-rigid subtyping over time⁶, and in a clinic-based cohort followed for
a mean of 6.2 years only 15–36% of patients classified tremor dominant at
baseline were still so classified at follow-up, depending on which of three
published algorithms was applied⁷. Those algorithms do not agree with one
another: prevalence of the tremor-dominant class ranged from 5% to 32% in the
same patients⁷, and in de novo patients three subtyping systems showed only
fair agreement (κ ≈ 0.36–0.38)⁸.

What this literature establishes is that the label *changes*. What it does not
establish is **why**. Every one of those studies measures change as a
comparison between a baseline and a later assessment, or as a shift in
prevalence across a cohort. That design cannot distinguish a patient whose
phenotype genuinely progressed from one whose single noisy assessment crossed a
threshold and crossed back, because both appear as a changed subtype.
Separating them requires three consecutive assessments and a statistic that
asks how often a change is undone — a quantity we did not find reported.

The question matters because the label is also routinely used as a prediction
target. Recent work has learned latent disease states from many clinical
measurements in this cohort⁹, and has predicted data-driven progression
subtypes from baseline data with external replication¹⁰; multistate Markov
models that treat a recorded state as a noisy reading of a latent one, through
an explicit misclassification matrix, are long established¹¹. But predictive
studies of Parkinson's disease subtypes generally report discrimination without
calibration — a pattern not specific to this field, calibration having appeared
in 12% of prediction studies against 27% for discrimination in a systematic
review of high-impact general medical journals¹² — and we did not find a study
that reports what a trivial rule achieves on the same task. Without that
comparison, an accuracy figure on a label this persistent is difficult to
interpret.

We therefore asked a question about the measurement rather than about any
model: **is longitudinal prediction of the motor subtype limited by the choice
of model family, or by the reliability of the label being predicted?** Using
3,361 consecutive labelled visit pairs contributed by 934 of 1,080 eligible
PPMI participants, we quantified how much observed subtype change reverses at
the very next assessment, and how sensitive the classification is to
perturbations smaller than its own measurement resolution. We then compared
nine model specifications spanning five families — observed-label Markov
chains, latent-state hidden Markov models, a hierarchical subscore mixed model,
Gaussian processes on the subscores, and a time-varying spline model — under
one evaluation protocol, on one set of prediction opportunities, against one
parameter-free baseline.

This study proposes no new model. Its unit of contribution is a protocol and a
set of measurements, and its central result is a convergence: a finding about
what the data will support rather than about what any particular method can do.

---

# §A — Claim-to-source map

Every numbered citation, and the sentence it supports.

| # | Reference | What it is cited for | Verified |
|---|---|---|---|
| 1 | Stebbins 2013, *Mov Disord* 28:668–670, doi:10.1002/mds.25383 | The classification rule, its item sets and thresholds | Crossref |
| 2 | Mestre 2021, *J Parkinsons Dis* 11:395–404, doi:10.3233/JPD-202472 | Subtype systems are numerous and were appraised; longitudinal stability was not assessed for most | Crossref + abstract |
| 3 | Kotagal 2016, *Ann Clin Transl Neurol* 3:473–477, doi:10.1002/acn3.312 | Whether PIGD is a distinct phenotype is contested | Crossref + abstract |
| 4 | Simuni 2016, *Parkinsonism Relat Disord* 28:62–67, PMID 27132498 | 39% of PIGD and 18% of TD changed subtype within 12 months, PPMI de novo | Europe PMC abstract |
| 5 | Lee 2019, *Front Neurol* 10:471, PMID 31133973 | TD 68.0% → 44.6%, PIGD 23.1% → 44.2% over four years, PPMI | Full text |
| 6 | Erro 2019, *Eur J Neurol* 26:1212–1218, doi:10.1111/ene.13968 | PIGD and akinetic-rigid subtyping are both unstable over time | Crossref metadata only |
| 7 | von Coelln 2021, *Parkinsonism Relat Disord* 88:13–18, PMID 34091412 | 15–36% of baseline TD still TD at 6.2 years; TD prevalence 5–32% across three algorithms | Europe PMC abstract |
| 8 | Ren 2021, *Front Neurosci* 15:637896, PMID 33732106 | Three subtyping systems agree only fairly, κ ≈ 0.36–0.38 | Europe PMC abstract |
| 9 | Severson 2021, *Lancet Digit Health* 3:e555–e564, PMID 34334334 | Latent disease states learned from many measurements in PPMI | Crossref + abstract |
| 10 | Dadu 2022, *npj Parkinsons Dis* 8:172, PMID 36526647 | Progression subtypes predicted from baseline, replicated externally | Crossref + abstract |
| 11 | Jackson 2003, *J R Stat Soc Ser D* 52:193–209, doi:10.1111/1467-9884.00351 | Multistate Markov models with classification error | Crossref |
| 12 | Bouwmeester 2012, *PLoS Med* 9:e1001221, PMID 22629234 | Calibration reported in 12% of prediction studies, discrimination in 27% | Europe PMC abstract |

> **Editorial — reference 6 is the weakest.** Erro 2019 is cited only for the
> general claim that subtyping schemes are unstable, which references 4, 5 and
> 7 also support. Its abstract was not retrieved; only Crossref metadata was
> verified. Either obtain the abstract before submission or drop the citation,
> which costs the argument nothing.

---

# §B — Claims deliberately not made

The two negative claims in paragraph four and five are the load-bearing ones,
and both are phrased as **"we did not find"** rather than "no study has". The
search terms behind them are recorded in `03_LITERATURE_REVIEW_20261104.md` §7
so a reader or reviewer can re-run them. That review was targeted and narrative,
not systematic, and the Introduction must not imply otherwise.

Specifically **not** claimed anywhere:

* that this is the first study of subtype instability — four cited studies
  precede it;
* that this is the first application of a hidden Markov model to disease
  progression — reference 11 is from 2003;
* that this is the first latent-state model in PPMI — reference 9 precedes it;
* that existing methods are inadequate, or that any method here is superior;
* that the label should be abandoned, which the results do not support.

The phrase "a quantity we did not find reported" is doing deliberate work: it
is weaker than a priority claim and is the strongest form the evidence allows.

---

# §C — Open items

1. **Reference 6** — obtain Erro 2019's abstract or drop it (above).
2. **Paragraph two** cites Mestre 2021 for the claim that the label is used as
   a stratifier and as a longitudinal quantity. That is supported by the
   appraisal's scope, but a more direct citation — a trial or cohort study that
   uses subtype as a stratification variable — would be stronger. Worth one
   targeted search before submission.
3. **Citation numbering** is in order of first appearance in this section only.
   It must be renumbered once Methods, Results and Discussion are assembled,
   at which point von Elm 2007 and Collins 2024 enter.
4. **Word count** is 693 words of manuscript prose, which is appropriate for an
   npj Article and leaves room if a reviewer asks for more clinical framing.

---

# §D — Corrected during verification

Two errors in the first draft of the roadmap sentence, both found by checking
the draft against the saved runs rather than against the outline:

1. **Denominator.** The draft read "1,080 patients and 3,361 consecutive
   labelled visit pairs", which pairs the full eligible cohort with a pair
   count that only a subset produced. `outputs/panel_summary.txt` gives 1,080
   eligible participants; `outputs/model/report.txt` and
   `outputs/patient_level/report.txt` both give 3,361 pairs, and Methods M3
   records that **934 of 1,080** contributed at least one prediction
   opportunity. Corrected to attribute the pairs to the 934.

2. **Fifth family.** The draft listed "a continuous multi-domain
   representation" as the fifth family. That is the clinical-profile model,
   which is a separate analysis (R7) and **not** one of the nine harmonised
   specifications. `harmonised_metrics.csv` gives the five non-baseline
   families as observed-label chain, latent-state HMM, subscore LMM, subscore
   GP and time-varying LMM. Corrected to "a time-varying spline model", which
   now matches R3 word for word.

The nine/five counts themselves check out against
`experiments/20261103T202632Z_e8_harmonised_family_comparison/harmonised_metrics.csv`:
14 rows, of which 5 are baselines, leaving 9 fitted specifications across the
5 families above.
