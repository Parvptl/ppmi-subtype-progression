# Manuscript review checklist

**Date** 2026-11-04 · **For** `manuscript/npj/MANUSCRIPT_REVIEW_DRAFT.pdf`
(25 pages) · **Source** `manuscript/npj/MANUSCRIPT_FULL_20261104.md`

What is settled, and what still needs your decision. Nothing here is a
scientific defect in the draft; the open items are declarations, allocation
choices and two questions I deliberately did not answer for you.

---

## 1. Complete and verified

| Item | State |
|---|---|
| Title and authors | Professor-approved title; four authors, two affiliations, as supplied |
| Abstract, Introduction, Results, Discussion, Methods | Complete; every headline figure re-derived from its saved run |
| Figures 1–5 (six panels) | All drawn, inspected as rendered pages, journal-spec |
| Tables 1–5 | All formatted; values carried through unchanged |
| References | 12 entries, each with a resolvable DOI, each cited, numbering complete |
| O-4 threshold sensitivity | Run, saved, 23/23 checks; Methods and Results both carry it |
| Five-state HMM convergence | Disclosed in Methods and Results |
| Growth model | Described in Methods; explicitly reports no result |
| Clustering investigation | Excluded throughout; verified absent from the PDF |
| Layout | No empty, sparse or orphaned pages; page numbers and running head present |

---

## 2. Needs your confirmation — declarations

**None of these exists anywhere in the project.** I have said so in the
manuscript rather than inventing anything; each appears as *"To be confirmed
by the authors"* with a sentence on what is required.

| Declaration | What is needed |
|---|---|
| **Ethics** | The institutional basis for secondary analysis of de-identified PPMI data — approval, exemption, or a determination that review is not required — plus the Data Use Agreement reference and its holder |
| **Funding** | Every source with grant numbers, or an explicit confirmation that none was received |
| **Competing interests** | A statement from each author individually |
| **Author contributions** | A CRediT statement per author |
| **PPMI acknowledgement** | PPMI's current required wording, verbatim. **Do not paraphrase** — it names sponsors and funding partners and is updated. `report/CARE_AI2026_Abstract.tex:109` carries your own TODO for it |
| **Corresponding author** | Name, institutional email, ORCIDs |
| **Repository hosting** | The URL, and whether the code will be public or deposited with a DOI. The code itself is complete and version-controlled; only hosting is open |

---

## 3. Decisions I did not make for you

### 3.1 Main paper versus supplementary

Everything substantive is currently in the main text. Candidates for
supplementary, in the order I would consider them:

| Analysis | Case for moving | Case for keeping |
|---|---|---|
| **R6** history-window and retraining | Two null results; the finding is "neither helped" | It forecloses an obvious reviewer question |
| **R8** per-participant composition | Explicitly exploratory, defined post hoc | It shows the persistence rule reproduces the same gradient, which is the paper's thesis in miniature |
| **R5** two-class secondary analysis | Derivative of R3 | It carries the calibration and AUC contrasts, which R3 cannot |
| Growth model Methods | Reports no result | Records what was actually fitted |

I would keep R5 and R8 and move R6. **That is a judgement, not a finding** —
it is yours to make.

### 3.2 Two questions left open deliberately

**Model-family comparability.** The harmonised comparison puts all nine
specifications on one decision rule, evaluation set and canonical reference,
which is what makes the 1.22-point spread meaningful. It does **not** put them
on one validation scheme — the hidden Markov model is leave-one-participant-out
and the clinical representation is five-fold. The manuscript states this
wherever the AUC contrast appears. Whether that is sufficient for the headline
convergence claim is a scientific judgement I have left to you.

**Five-state HMM.** It converged in 404 of 934 folds and 59.2% of its
predictions come from fits that did not converge. The manuscript discloses
this and notes the model is interior to the range, so dropping it leaves the
1.22-point spread unchanged. Whether to report it at all, rather than merely
disclose it, is open.

---

## 4. Items carried forward, unresolved

| Item | Status |
|---|---|
| **E-7 reporting checklists** | Not run, by your instruction. STROBE and TRIPOD+AI are therefore **deliberately uncited**, since citing a standard the manuscript has not been assessed against would assert a compliance that does not exist. Many journals request the completed checklist at submission |
| **Reference 6 (Erro 2019)** | Verified only through Crossref metadata; the abstract was never retrieved. It supports a claim references 4, 5 and 7 already carry. Obtain it or drop it |
| **R7 heading** | "A continuous representation carries information the label does not" asserts more than its properly hedged body |
| **Precision rendering** | Persistence accuracy appears as 0.7227 in Results and 0.723 in the Discussion; two R7 values sit on exact three-decimal ties |
| **Introduction roadmap** | Foreshadows R2 and R3 but not R5–R8, including R7, which the Discussion leans on hardest |

---

## 5. One thing to look at specifically

**Figure 3 previously conflated two different persistence rules.** Every
model's connector was drawn from recorded-label persistence (0.7227), while
the figure also plotted subscore LOCF persistence (0.7453) as an ordinary row.
A reader could have taken either as the reference.

Corrected in this draft: connectors are drawn **only** for the nine fitted
specifications and **only** to the canonical reference; both rules are marked
with distinct lines and named; and the caption now states explicitly that they
are not interchangeable and that no contrast is taken against the LOCF rule.
The subtitle changed from "one baseline" to "one canonical reference" to match.

Worth checking that the figure, its caption, R3 and Table 3 now read
consistently to you, because this is the one place where a presentational
choice could have implied an advantage the analysis does not support.

---

## 6. What this draft is not

It is a complete working manuscript for scientific review. It is **not**
submission-ready: seven declarations are outstanding, no reporting checklist
has been run, and the main-versus-supplementary allocation is undecided. No
claim of readiness is made anywhere in the PDF.
