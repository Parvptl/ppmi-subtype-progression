# Stage 3 — literature review, verified

**Date** 2026-11-04 · **Follows** `01_STAGE1_RESEARCH_AUDIT_20261104.md` and
`02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md`

Every reference below was resolved through the **Crossref REST API** by DOI and,
where an abstract was needed, through **Europe PMC**. Nothing is cited from a
search-result snippet alone. No citation here was taken on trust from the
project's existing documents; Stebbins and Severson, which the 21-October
manuscript already cites, were re-verified independently and both are correct
as printed there.

§8 states plainly what this search could and could not reach. **It is a
targeted narrative review, not a systematic one**, and the gap statement in §6
is phrased as "I did not find", never as "no study has".

---

## 1. Evidence table

Confidence reflects what I actually read: **High** = abstract or full text read
and the number quoted appears in it; **Moderate** = abstract read, claim is a
summary rather than a quoted figure; **Metadata only** = bibliographic record
verified, content not read.

| # | Claim | Source | Population | Method | Evidence | DOI / PMID | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | The TD/PIGD label is a thresholded ratio of two MDS-UPDRS item means | Stebbins 2013 | — (methodological) | Derivation against UPDRS | Defines the 1.15 / 0.90 rule this project implements | 10.1002/mds.25383 | Metadata only (rule independently confirmed by the project's 100% agreement with PPMI's own derivation) |
| 2 | Subtype classification is unstable within one year, especially for PIGD | Simuni 2016 | PPMI de novo, 423 recruited / 320 analysed | Baseline → 12-month reclassification | 39% of PIGD and 18% of TD changed subtype; PIGD→TD 29%, PIGD→Ind 11%, TD→PIGD 10%, TD→Ind 8% | 10.1016/j.parkreldis.2016.04.027 / 27132498 | **High** |
| 3 | Baseline subtype distribution in PPMI de novo PD | Simuni 2016 | as above | Cross-sectional | TD 228 (71%), PIGD 56 (18%), Indeterminate 36 (11%) | as above | **High** |
| 4 | Dopaminergic therapy does not drive subtype change | Simuni 2016 | as above | Comparison by treatment | p = 0.59 | as above | **High** |
| 5 | Subtypes are inconsistent across algorithms and unstable over years | von Coelln 2021 | Clinic-based longitudinal PD database; 1,185 cross-sectional, 178 longitudinal, mean follow-up 6.2 ± 1.0 y | Three published algorithms compared | Prevalence varies by algorithm: TD 5–32%, Ind/mixed 9–31%, PIGD 59–75%. Only **15–36%** of baseline TD remain TD at follow-up; 46–50% shift to PIGD | 10.1016/j.parkreldis.2021.05.016 / 34091412 | **High** |
| 6 | The TD proportion falls and PIGD rises over 4 years in PPMI | Lee 2019 | PPMI, 325 drug-naïve at baseline; baseline, 1, 2, 4 y | Descriptive transition matrix | TD 68.0% → 44.6%; PIGD 23.1% → 44.2%. Baseline TD at 4 y: **58.3% TD**, 12.8% Ind, 29.1% PIGD. Baseline PIGD: 71.4% PIGD, 17.8% TD | 10.3389/fneur.2019.00471 / 31133973 | **High** |
| 7 | Lee 2019 uses **different thresholds** from this project | Lee 2019 | as above | TD ratio ≥ 1.5, PIGD ≤ 1.0, Ind between | The older UPDRS/Jankovic cut-offs, not Stebbins 1.15 / 0.90 | as above | **High** |
| 8 | Three subtyping systems agree only fairly in de novo PD | Ren 2021 | Single-centre, 283 de novo at baseline, 79 followed, re-assessed at 1 month | Jankovic vs Schiess vs Kang, Cohen's κ | κ = 0.383 ± 0.044, 0.360 ± 0.042, 0.368 ± 0.038; Jankovic least stable | 10.3389/fnins.2021.637896 / 33732106 | **High** |
| 9 | PIGD and akinetic-rigid subtyping are both unstable over time | Erro 2019 | PD cohort, longitudinal | Two subtyping schemes compared over time | Reported as showing substantial switching | 10.1111/ene.13968 | **Moderate** (metadata verified; abstract not retrieved) |
| 10 | The PIGD construct's validity is itself contested | Kotagal 2016 | — (review) | Narrative review of data-driven clustering | Data-driven analyses recover a tremor cluster but not reliably a distinct PIGD cluster | 10.1002/acn3.312 | **Moderate** |
| 11 | Longitudinal stability was **not assessed** for most published subtype systems | Mestre 2021 | — (critical appraisal, 24 authors, MDS-endorsed) | Systematic appraisal of subtype classifications | Whether subtypes are stable "could not be evaluated … since longitudinal differences were not assessed in the original descriptions" | 10.3233/JPD-202472 | **Moderate** |
| 12 | Multistate Markov models with an explicit misclassification matrix are an established method | Jackson 2003 | — (methodological) | CT Markov chain + classification-error matrix, ML estimation | The canonical framework for exactly the model this project fits; assumes time-homogeneous rates between observations | 10.1111/1467-9884.00351 | **Moderate** |
| 13 | A personalised input-output HMM on PPMI yields eight disease states | Severson 2021 | PPMI + PDBP | Contrastive LVM + personalised IOHMM | Eight states differentiated by function, tremor, bradykinesia, neuropsychiatric measures; no observation-misclassification model | 10.1016/S2589-7500(21)00101-1 / 34334334 | **Moderate** |
| 14 | ML can predict data-driven *progression-rate* subtypes from baseline | Dadu 2022 | PPMI n = 294 discovery, PDBP n = 263 validation | Unsupervised + supervised ML, external cohort | Three progression subtypes; 5-year projection AUC 0.87–0.95. **No calibration reported in the abstract** | 10.1038/s41531-022-00439-z / 36526647 | **Moderate** |
| 15 | Calibration is the most under-reported element of prediction research | Bouwmeester 2012 | 71 prediction papers, 6 high-impact general medical journals, 2008 | Systematic review of reporting | Calibration reported in **12%** of studies, discrimination in **27%** | 10.1371/journal.pmed.1001221 / 22629234 | **High** |
| 16 | Current reporting guidance for prediction models | Collins 2024 (TRIPOD+AI) | — (guideline) | Consensus statement | The standard this manuscript must be checked against | 10.1136/bmj-2023-078378 | **High** |
| 17 | Current reporting guidance for observational studies | von Elm 2007 (STROBE) | — (guideline) | Consensus statement | The standard for the cohort component | 10.1016/S0140-6736(07)61602-X | **High** |

No Crossref record carried a retraction, correction or update relation. That is
weaker than a dedicated retraction check and is stated as such in §8.

---

## 2. What the literature establishes

### 2.1 Subtype instability is known, and consistently found

Four independent groups, four cohorts, four designs, same conclusion. Simuni
(PPMI, 1 year), Lee (PPMI, 4 years), Erro (2 schemes over time), von Coelln
(clinic cohort, 6.2 years). **This project's central descriptive claim is not
novel** and the manuscript must not present it as such.

### 2.2 But the instability is measured as *net drift*, not as *noise*

This is the opening the project actually has. Every study above measures
subtype change as a **baseline-to-follow-up** comparison or a **prevalence
shift**. That design cannot separate two very different things:

* a patient whose phenotype genuinely progressed, and
* a patient whose single noisy assessment crossed a threshold and crossed back.

Both appear as "changed subtype". Distinguishing them needs **three**
consecutive assessments, which is what a reversion rate uses and what none of
the retrieved studies reports. Lee 2019 comes closest — it has four
assessments — but reports only baseline-to-4-year transitions, not
visit-to-visit reversion.

### 2.3 Algorithm dependence is documented; within-algorithm fragility is not

von Coelln shows three algorithms give TD prevalence anywhere from 5% to 32%,
and Ren puts the agreement between three schemes at κ ≈ 0.36–0.38 — only fair.
Both are **between-algorithm** findings.

This project's threshold-sensitivity result is a **within-algorithm** version
of the same problem: moving each Stebbins threshold by about 0.02 reclassifies
**67.6%** of the Indeterminate class (§13 of the Stage-1 audit). I did not find
a published equivalent. This is a narrow but genuine addition.

### 2.4 The HMM machinery is standard; the application appears not to be

Jackson 2003 is the canonical multistate-Markov-with-misclassification
framework, and is the correct methodological citation for the project's CT-HMM —
the manuscript currently cites neither it nor any multistate-model literature,
which a methods reviewer would notice immediately. Severson 2021 is the closest
PD application, but it discovers eight states from many measurements rather
than treating the TD/PIGD label as a noisy reading of a latent subtype, and it
has no misclassification matrix.

### 2.5 Predictive work targets different things, and does not report calibration

Dadu 2022 is the strongest retrieved predictive paper — PPMI discovery, PDBP
external validation, AUC 0.87–0.95 — but it predicts **model-generated
progression-rate clusters** from baseline, not the next recorded clinical
label, and its abstract reports no calibration. Bouwmeester 2012 shows that is
the norm rather than an oversight specific to PD: calibration appeared in 12%
of prediction studies against 27% for discrimination.

I found **no** study predicting the next recorded TD/PIGD label against a
persistence baseline. §8 states the search terms, so the claim can be checked.

---

## 3. External corroboration of this project's numbers

This is the most useful thing the review produced, and none of it was available
to the project before. These are independent published analyses, not
re-analyses of the project's own panel.

### 3.1 Baseline distribution — agreement to within one percentage point

| | TD | Indeterminate | PIGD |
|---|---:|---:|---:|
| **This project** (1,076 baseline visits) | **71.7%** | **10.1%** | **18.1%** |
| Simuni 2016 (320 PPMI de novo) | 71% | 11% | 18% |

Two independent reconstructions of the TD/PIGD label from PPMI agree to within
a percentage point on all three classes. **This is a stronger validation of the
label pipeline than the project's existing 100% / 99.87% agreement checks**,
because those compare against PPMI's own derivation of the same formula from
the same items, whereas Simuni is a separate analysis by a separate group. The
manuscript should use it, and should stop calling the internal agreement checks
validation (Stage-1 audit §2.1a).

### 3.2 One-year change — the TD row matches almost exactly

| From TD | → TD | → Indeterminate | → PIGD |
|---|---:|---:|---:|
| **This project** (median interval 1.00 y) | **80.8%** | **8.0%** | **11.2%** |
| Simuni 2016 (12 months) | 82% | 8% | 10% |

| From PIGD | → PIGD | → Indeterminate | → TD |
|---|---:|---:|---:|
| **This project** | **70.2%** | **11.4%** | **18.4%** |
| Simuni 2016 (12 months) | 61% † | 11% | 29% |

† Simuni's abstract states that 39% of PIGD changed subtype, so 61% persisted, while its component figures (29% + 11%) sum to 40%. The one-point discrepancy is rounding in the source and is reproduced here rather than silently resolved.

The TD row agrees to within two points on every cell, including the split
between destinations. The Indeterminate share of PIGD change agrees exactly
(11%). The project finds **less** PIGD→TD reversion than Simuni (18.4% vs 29%),
which is the one real discrepancy and has a plausible explanation: Simuni
measures baseline→Y1 in a *de novo* cohort, where PIGD is newly assigned and
least settled, whereas this project pools all consecutive pairs across 14 years,
most of which come from later follow-up where PIGD is more established. The
project's own interval- and stage-stratified results support that reading and
should be cited when the discrepancy is discussed, rather than leaving it
unexplained.

### 3.3 Long-horizon stability — consistent with the PPMI study, not the clinic study

| | TD still TD |
|---|---|
| **This project**, >4-year interval stratum | **57.8%** (95% CI 43.2–71.9) |
| Lee 2019, PPMI, baseline → 4 years | 58.3% |
| von Coelln 2021, clinic cohort, mean 6.2 years | 15–36% |
| **This project**, naive compounding of the per-visit matrix over 5 visits | 34.4% |

Lee's 4-year figure and this project's >4-year stratum agree to within half a
percentage point — **despite Lee using the older ≥1.5 / ≤1.0 thresholds rather
than Stebbins 1.15 / 0.90.** That the two agree across a definitional difference
is worth a sentence in the Discussion.

von Coelln's 15–36% is much lower, and the cohorts differ in a way that
explains it: von Coelln's PIGD prevalence is **59–75%** against PPMI's ~25%, so
that is a substantially more advanced clinic population, and the 15–36% range
spans three different algorithms. The honest statement is that this project's
long-horizon stability matches the PPMI-based estimate and not the clinic-based
one, and that cohort composition is the likely reason.

### 3.4 A discrepancy the comparison exposes — in the project, not the literature

Placing the project's five-year clinical output table beside the published
figures showed that the table is not the quantity it is presented as.

`hmm.clinical_table` returns **latent-to-latent** transition probabilities. Its
own docstring says so — *"rows are the patient's true current phenotype, not
their recorded label"* — but the README and the Weeks 5–7 report quote it as
a clinical statement: *"A patient currently Indeterminate has a 63% chance of
being PIGD within five years, against 19% for a patient currently TD."* A
clinician reading that would take it as a statement about a patient whose
**recorded** label is Indeterminate. It is not: the latent state is never
observed.

Recomputing from the saved generator and emission matrices:

| Five-year projection | TD → TD | TD → PIGD | Ind → PIGD |
|---|---:|---:|---:|
| **(A) latent → latent** — what is quoted | 0.660 | **0.191** | **0.633** |
| (B) latent → recorded label | 0.679 | 0.212 | 0.583 |
| **(C) recorded → recorded**, flat latent prior | 0.560 | **0.327** | 0.574 |
| **(C) recorded → recorded**, using the LOPO-fitted π | 0.649 | **0.240** | 0.500 |

The quantity a clinician would want is (C). It differs from the quoted figure
by **5 to 14 percentage points**, and moves in opposite directions for the two
rows: the TD→PIGD risk is *understated* (0.19 quoted against 0.24–0.33) and the
Indeterminate→PIGD risk *overstated* (0.63 quoted against 0.50–0.57). Note also
that (C) is sensitive to the initial distribution π, which `outputs/model/`
does not save — the two rows use a flat prior and the π recorded by the LOPO
fit, and they differ by nine points on TD→PIGD.

**Three consequences for the manuscript.** The clinical table must be labelled
as a latent-state quantity or recomputed through the emission matrix; π must be
saved so the recomputation is reproducible; and the emission-adjusted TD→TD
figure (0.56–0.65 at five years) is what should be compared with Lee's 58.3%,
which it matches well. This is a presentation defect, not a modelling error —
the code is correct and its docstring is honest.

---

## 4. Research gap

Supported by §2 and §3, and phrased so each clause can be checked.

**What is established.** The TD/PIGD label changes frequently over follow-up;
the direction of net drift is TD → PIGD; different subtyping algorithms
disagree substantially; the PIGD construct's validity is contested; and for most
published subtype systems, longitudinal stability was never assessed at all.

**What is not established, and what this project can add.**

1. **Instability has been measured as net drift, not as immediate reversion.**
   No retrieved study reports what share of observed subtype changes reverse at
   the very next assessment — the statistic that separates progression from
   measurement noise, and which requires three consecutive assessments. This
   project measures 35.2% (95% CI 31.8–38.8 from the patient-clustered
   bootstrap in E-2).
2. **The label has not, in the retrieved literature, been modelled as a noisy
   reading of a latent subtype.** Jackson's misclassification framework is
   standard and Severson's PPMI work is adjacent, but I found no application of
   an explicit misclassification matrix to the TD/PIGD label.
3. **Predictive work on PD subtypes does not report a trivial baseline.** The
   retrieved studies predict model-generated clusters from baseline data. I
   found none predicting the next recorded label, and none reporting what a
   persistence rule achieves — which matters here because persistence reaches
   0.8336 against the best model's 0.8369.
4. **Calibration is not reported.** Consistent with Bouwmeester's finding that
   calibration appears in 12% of prediction studies generally.
5. **Within-algorithm threshold fragility is undocumented.** The literature
   compares algorithms; it does not perturb one algorithm's thresholds. Moving
   the Stebbins cut-offs by ~0.02 reclassifies two thirds of the Indeterminate
   class.
6. **No retrieved study compares several model families under one protocol on
   one evaluation set.** This project compares five, and finds they converge —
   which is only interpretable because the protocol is held fixed.

**The defensible framing.** Not "we predict subtype better", but: *the published
literature establishes that this label is unstable; this work measures how much
of that instability is immediate reversion rather than progression, shows that
five model families spanning very different assumptions converge within a few
points of a persistence rule, and concludes that the binding constraint is the
reliability of the label rather than the capacity of the model.* That is the
title the project has already written for itself, and the literature supports
it.

**Phrasings to avoid**, per §21 of the brief and warranted here: this is not the
first study of subtype instability (Simuni, Lee, Erro, von Coelln precede it),
not the first HMM in disease progression (Jackson 2003), and not the first
latent-state model in PPMI (Severson 2021).

---

## 5. Where each finding belongs in the manuscript

| Manuscript section | Citations |
|---|---|
| Introduction — the label and its definition | Stebbins 2013 |
| Introduction — instability is known | Simuni 2016, Lee 2019, Erro 2019, von Coelln 2021 |
| Introduction — algorithms disagree, construct contested | von Coelln 2021, Ren 2021, Kotagal 2016, Mestre 2021 |
| Introduction — the gap | Mestre 2021 (stability not assessed), Bouwmeester 2012 (calibration under-reported) |
| Methods — CT-HMM with misclassification | **Jackson 2003** (currently uncited; must be added) |
| Methods — reporting standards | von Elm 2007, Collins 2024 |
| Results R1 — external corroboration | Simuni 2016 (§3.1, §3.2), Lee 2019 (§3.3) |
| Discussion — comparison with literature | von Coelln 2021 (cohort difference), Severson 2021, Dadu 2022 |

---

## 6. Verified references

1. **Stebbins GT, Goetz CG, Burn DJ, Jankovic J, Khoo TK, Tilley BC.** How to identify tremor dominant and postural instability/gait difficulty groups with the Movement Disorder Society Unified Parkinson's Disease Rating Scale: comparison with the Unified Parkinson's Disease Rating Scale. *Mov Disord* 2013;28(5):668–670. doi:10.1002/mds.25383
2. **Simuni T, Caspell-Garcia C, Coffey C, Lasch S, Tanner C, Marek K.** How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? *Parkinsonism Relat Disord* 2016;28:62–67. doi:10.1016/j.parkreldis.2016.04.027. PMID:27132498
3. **Lee JW, Song YS, Kim H, Ku BD, Lee WW.** Alteration of tremor dominant and postural instability gait difficulty subtypes during the progression of Parkinson's disease: analysis of the PPMI cohort. *Front Neurol* 2019;10:471. doi:10.3389/fneur.2019.00471. PMID:31133973
4. **von Coelln R, Gruber-Baldini AL, Reich SG, Armstrong MJ, Savitt JM, Shulman LM.** The inconsistency and instability of Parkinson's disease motor subtypes. *Parkinsonism Relat Disord* 2021;88:13–18. doi:10.1016/j.parkreldis.2021.05.016. PMID:34091412
5. **Erro R, Picillo M, Amboni M, et al.** Comparing postural instability and gait disorder and akinetic-rigid subtyping of Parkinson disease and their stability over time. *Eur J Neurol* 2019;26:1212–1218. doi:10.1111/ene.13968
6. **Ren J, Pan C, Li Y, Li L, Hua P, Xu L, Zhang L, Zhang W, Xu P, Liu W.** Consistency and stability of motor subtype classifications in patients with de novo Parkinson's disease. *Front Neurosci* 2021;15:637896. doi:10.3389/fnins.2021.637896. PMID:33732106
7. **Kotagal V.** Is PIGD a legitimate motor subtype in Parkinson disease? *Ann Clin Transl Neurol* 2016;3(6):473–477. doi:10.1002/acn3.312
8. **Mestre TA, Fereshtehnejad SM, Berg D, et al.** Parkinson's disease subtypes: critical appraisal and recommendations. *J Parkinsons Dis* 2021;11(2):395–404. doi:10.3233/JPD-202472
9. **Jackson CH, Sharples LD, Thompson SG, Duffy SW, Couto E.** Multistate Markov models for disease progression with classification error. *J R Stat Soc Ser D (The Statistician)* 2003;52(2):193–209. doi:10.1111/1467-9884.00351
10. **Severson KA, Chahine LM, Smolensky LA, Dhuliawala M, Frasier M, Ng K, Ghosh S, Hu J.** Discovery of Parkinson's disease states and disease progression modelling: a longitudinal data study using machine learning. *Lancet Digit Health* 2021;3(9):e555–e564. doi:10.1016/S2589-7500(21)00101-1. PMID:34334334
11. **Dadu A, Satone V, Kaur R, et al.** Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. *npj Parkinsons Dis* 2022;8(1):172. doi:10.1038/s41531-022-00439-z. PMID:36526647
12. **Bouwmeester W, Zuithoff NPA, Mallett S, Geerlings MI, Vergouwe Y, Steyerberg EW, Altman DG, Moons KGM.** Reporting and methods in clinical prediction research: a systematic review. *PLoS Med* 2012;9(5):e1001221. doi:10.1371/journal.pmed.1001221. PMID:22629234
13. **Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al.** TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378
14. **von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP.** The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. *Lancet* 2007;370(9596):1453–1457. doi:10.1016/S0140-6736(07)61602-X

Carried over from the 21-October manuscript and **not re-verified in this
pass**, so they still need checking before submission: Severson 2020 (PMLR
126:309–330), Severson/Ghosh/Ng 2019 (AAAI), Goetz 2008 (MDS-UPDRS),
Tomlinson 2010 (LEDD), Marek 2011 (PPMI), Bengio & Frasconi 1995.

---

## 7. What this review did not find, and the exact terms used

Stated so the negative claims in §4 can be audited rather than taken on trust.

| Searched for | Terms used | Outcome |
|---|---|---|
| Reversion rate between consecutive assessments | "TD PIGD motor subtype instability longitudinal change PPMI proportion shift over time"; "Erro Nutt motor subtypes Parkinson transient longitudinal reclassification critique" | Not found. All retrieved studies report baseline→follow-up change or prevalence shift |
| HMM with misclassification applied to PD motor subtype | "hidden Markov model Parkinson's disease progression latent state misclassification multistate model longitudinal" | Not found. Jackson 2003 is the generic framework; Severson 2021 is the nearest PD application and has no misclassification layer |
| Persistence / trivial baseline in PD subtype prediction | "machine learning predict Parkinson motor subtype conversion next visit calibration baseline comparison validation PPMI" | Not found |
| Calibration reported in PD subtype prediction | same as above | Not found in the retrieved studies; Bouwmeester 2012 shows this is the general pattern |
| Within-algorithm threshold sensitivity | "Parkinson disease motor subtype TD PIGD classification reproducibility measurement error threshold criticism review" | Not found. The retrieved work compares algorithms, not perturbations of one |

---

## 8. Limitations of this review

1. **Not systematic.** No PRISMA protocol, no pre-registered search string, no
   dual screening, no formal inclusion criteria. It is a targeted narrative
   review aimed at the six specific claims the Stage-1 audit said a gap
   statement would rest on.
2. **Abstracts, not full texts, for most sources.** Simuni, von Coelln, Erro
   and Kotagal are subscription-only. Their numbers come from Europe PMC
   abstract records, which I read directly, not from the methods sections. Any
   figure I quote from them is an abstract figure.
3. **Erro 2019 is the weakest entry.** Bibliographic metadata verified via
   Crossref; the abstract was not retrieved. It is cited only for the general
   claim that subtyping schemes are unstable, which three other sources also
   support.
4. **Retraction checking is metadata-level only.** No Crossref record carried a
   retraction, correction or update relation. I did not check Retraction Watch
   or the journals' own pages.
5. **PubMed could not be fetched directly** (cookie wall), so all verification
   routed through the Crossref REST API and Europe PMC. Google Scholar was not
   searchable through these tools.
6. **English-language, US-region search.** Non-English and regionally indexed
   work may be missed.
7. **Negative findings are "not retrieved", not "non-existent".** §7 gives the
   exact terms so the claims can be re-run. A domain expert's hand-search would
   be a reasonable next check, particularly for the reversion-rate claim, which
   is the one doing the most work in the gap statement.
8. **One comparison in §3 is mine, not published.** The compounding of the
   per-visit matrix over five visits (§3.3) and the emission-adjusted
   projections (§3.4) are my own calculations from the project's saved outputs,
   shown so the comparison with Lee and von Coelln is like-for-like. They are
   arithmetic on existing results, not new experiments, and the §3.4 figures
   depend on an initial distribution the project does not save.
