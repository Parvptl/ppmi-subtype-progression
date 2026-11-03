# Stage 10 — Title and Abstract, drafted

**Date** 2026-11-04 · **Target** npj Parkinson's Disease, Article · **Status**
first full draft

Drafted last, as the brief requires, so that every figure quoted here is one
already verified in Results and interpreted in the Discussion. **No quantity
appears here that does not appear in R1–R8.**

Constraints applied, as recorded in `04_SCIENTIFIC_POSITIONING_20261104.md` §7:
abstract **unstructured, ≤150 words**; title **≤15 words, no punctuation,
idioms or puns**. Both were taken from Nature's content-types page reached
through search — **the live page was behind an authentication redirect and
could not be fetched directly, so both must be re-checked at submission.**

---

## Title

**Recommended:**

> Label reliability rather than model capacity limits longitudinal prediction
> of Parkinson's disease motor subtypes

14 words, no punctuation. It names the finding rather than the method, which
suits a paper whose unit of contribution is a protocol and a set of
measurements.

**Alternatives**, if a more neutral framing is wanted:

| | Title | Words |
|---|---|---|
| B | Five model families converge on a persistence baseline for Parkinson's disease motor subtype prediction | 14 |
| C | Immediate reversion of Parkinson's disease motor subtype change across 3361 consecutive labelled visit pairs | 14 |

B foregrounds the comparison and reads as the more conventional methods paper.
C foregrounds the measurement and is the most conservative of the three, but it
drops the conclusion entirely.

> **Editorial — the working title needed two corrections.** The title carried
> forward from the CARE AI 2026 abstract was *"Label reliability, not model
> capacity, limits longitudinal prediction of Parkinson's disease motor
> subtypes"*. First, **it contains two commas**, which conflicts with the npj
> title rule as recorded at Stage 4; the recommended version replaces the
> appositive with "rather than" and is punctuation-free with the meaning
> intact. Second, its length was recorded as "fifteen words" in
> `04_SCIENTIFIC_POSITIONING` and "14 words" in `06_MANUSCRIPT_OUTLINE`; it is
> **13**. Neither error changed a conclusion, but both were carried in text
> that claimed compliance with a limit.

---

## Abstract

The tremor-dominant versus postural-instability-gait-difficulty subtype, a
thresholded ratio of two MDS-UPDRS item means, is widely used as a longitudinal
endpoint in Parkinson's disease. Among 1,080 Parkinson's Progression Markers
Initiative participants, 934 contributed 3,361 consecutive labelled
transitions. 35.2% of observed subtype changes reversed at the next visit. Nine
specifications spanning five model families, from Markov chains to Gaussian
processes, fell within 1.2 accuracy points of one another under one decision
rule, evaluation set and baseline. All exceeded a parameter-free persistence
rule by 2.3 to 3.5 points, but 94% of the latent-state model's margin arose on
the 9.9% of transitions recorded indeterminate; no fitted model recovered that
class, and persistence achieved the highest macro-F1. A continuous nine-domain
profile of the same examinations discriminated subsequently recorded Hoehn and
Yahr stage 3 or worse, and mild cognitive impairment, better than the label.
The binding constraint was label reliability, not model capacity.

**147 words** — three under the limit.

---

# §A — Every abstract quantity and its source

| Quantity | Value | Source |
|---|---|---|
| Participants in cohort | 1,080 | `outputs/panel_summary.txt`; R1 |
| Participants contributing transitions | 934 | R1; Methods M3 |
| Consecutive labelled transitions | 3,361 | `outputs/model/report.txt`, `outputs/patient_level/report.txt`; R1 |
| Immediate reversion | 35.2% | E-2 `other_quantities.csv`; R2 |
| Spread across specifications | 1.2 accuracy points | E-8 `harmonised_metrics.csv`; R3 |
| Margin over persistence | 2.3 to 3.5 points | E-8 `paired_contrasts.csv` (0.0226–0.0348); R3 |
| Share of latent-state margin in one stratum | 94% | E-8 `margin_decomposition.csv` (94.3%); R4 |
| Indeterminate share of transitions | 9.9% | R4; Methods M12b (332 of 3,361) |
| No fitted model recovered indeterminate | — | R3 |
| Persistence has highest macro-F1 | 0.5559 | E-8 `harmonised_metrics.csv`; R3 |
| Profile beats label on H&Y ≥ 3 and MCI | +0.105, +0.161 | R7 `outcome_associations.csv` |

**Deliberately excluded from the abstract**, per the outline's instruction: any
claim of clinical utility; any "first"; and the two-class 0.84 accuracy figure,
which must never appear without its three-class 0.75 counterpart.

---

# §B — A false claim caught before it reached the abstract

The first draft of the abstract read *"fell within 1.2 accuracy points of one
another under one leave-one-participant-out protocol."* **That is not true.**

Methods M12a harmonises the **decision rule, the evaluation set and the
baseline**, by re-scoring each family's *saved predictions* onto a single
footing via Monte Carlo through the unmodified classification rule. It does
**not** impose a single validation scheme: each family's predictions come from
its own run, and the schemes genuinely differ — the Discussion's limitations
already disclose leave-one-participant-out for the hidden Markov model against
five-fold cross-validation for the representation.

`scripts/79_harmonised_family_comparison.py` contains no fold or
leave-one-out logic at all, which is what exposed it. The clause now reads
"under one decision rule, evaluation set and baseline", matching R3 word for
word. Had it survived, the abstract would have asserted a methodological
uniformity the study does not have — the single most consequential kind of
error available in 150 words.

---

# §C — Open items

1. **Both limits are unverified.** The 150-word and 15-word rules were read
   from a search result, not from the live submission page. Re-check before
   submission; if the abstract limit is lower, §A's table shows which
   quantities are load-bearing (reversion, spread, margin localisation) and
   which can go (the 1,080/934 split, the macro-F1 value).
2. **Hyphenated compounds.** "tremor-dominant",
   "postural-instability-gait-difficulty" and "nine-domain" are counted as one
   word each here. A submission system that splits on hyphens would score this
   abstract at 152. The margin is three words, so if their counter disagrees,
   cut "and mild cognitive impairment," first — it is the least load-bearing
   clause.
3. **Title choice is the authors'.** The recommendation states the conclusion
   in the title. Some reviewers read that as overreach even when the body
   supports it; alternative C is available if a conservative framing is
   preferred.
