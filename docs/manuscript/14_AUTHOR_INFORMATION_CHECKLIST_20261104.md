# Author-information checklist

**Date** 2026-11-04 · **For** `manuscript/npj/MANUSCRIPT_FULL_20261104.md` ·
**Target** npj Parkinson's Disease

Information the manuscript cannot supply for itself. **Nothing here is drafted
or guessed** — in particular no acknowledgement wording, no competing-interests
statement and no ethics language has been written, because each would be a
factual claim about the authors or their institution that the repository does
not support.

Two parts: **§1** is what the repository does establish, with its source, so
the authors are confirming rather than recalling. **§2** is what only the
authors can provide.

---

## 1. Established from repository materials

Each row is a fact with a file behind it. These still want author
**confirmation** — a repository is evidence of what was recorded, not of what
is correct — but they do not need to be supplied from scratch.

| Item | Value in the repository | Source |
|---|---|---|
| **Title** | Label Reliability, Not Model Capacity, Limits Longitudinal Prediction of Parkinson's Disease Motor Subtypes | `report/CARE_AI2026_Abstract.tex` L17–18 |
| **Author list and order** | Parv Patel, Mayank Tewatia, and Dr. Sahasranand K. R. | same, L19 |
| **Affiliation** | Indian Institute of Technology Palakkad, India — given as a *single* affiliation covering all three authors | same, L20 |
| **Data source** | Parkinson's Progression Markers Initiative (PPMI), `ppmi-info.org` | same, L108 |
| **Git authorship** | Mayank Tewatia (24 commits); Parv (23 as `Parv`, 9 as `Parvptl`, same email — 32 total) | `git log` |
| **Code is version-controlled** | All analysis code, including the Gaussian-process modules, is tracked; the GP implementation was added 2026-09-30 in `dada934`, already on `origin/main` | `git ls-files`, `git log` |
| **Frozen panel checksum** | Recorded, so a reader with PPMI access can reconstruct the identical panel | run manifests; `asset_manifest.json` |

> **Note on the title.** The manuscript uses a punctuation-free 14-word variant
> of this title, because the abstract's version carries two commas and the npj
> rule as recorded prohibits title punctuation. Two neutral alternatives are in
> `11_ABSTRACT_TITLE_DRAFT_20261104.md`. Choosing among them is an author
> decision, not a defect.

---

## 2. Requires author confirmation

Seven items. All seven are **blocking for submission**; none is a scientific
defect, and none can be resolved from repository materials.

### 2.1 Ethics approval or exemption — **blocking**

**Needed.** The institutional position on secondary analysis of PPMI data:
either an approval reference, a formal exemption or
not-human-subjects-research determination, or a statement that the
institution's policy does not require review for de-identified secondary
analysis under a Data Use Agreement. Also the PPMI Data Use Agreement
reference under which access was granted, and who holds it.

**Repository status.** **Nothing.** A repository-wide search for *ethics*,
*IRB*, *consent*, *approval* and *exemption* returns no record outside the
placeholder markers this work added.

**Why it matters.** npj Parkinson's Disease requires an ethics declaration.
Secondary analysis of a de-identified consortium cohort very often qualifies
for exemption, but **which** basis applies is an institutional fact, and
stating the wrong one is worse than stating none.

**Goes in.** `## Ethics declarations`, currently marked ABSENT at L914.

### 2.2 Funding — **blocking**

**Needed.** Every funding source supporting this work, with grant numbers, or
an explicit confirmation that none was received. If this is BTP or
coursework-associated work, whether the institution requires acknowledgement.

**Repository status.** **Nothing.** No grant number, funder or support
statement anywhere.

**Why it matters.** A declaration is mandatory. "No funding was received" is a
legitimate statement but is a **factual claim** that must come from the
authors — it cannot be inferred from the absence of records.

**Goes in.** `## Funding`, L918.

### 2.3 Competing interests — **blocking**

**Needed.** A statement from **each** author.

**Repository status.** **Nothing**, and nothing could be there.

**Why it matters.** The absence of a declaration must not be read as an absence
of conflicts, and no such statement has been drafted here for that reason.
Each author must declare individually; a blanket "the authors declare no
competing interests" written on their behalf would be fabricated.

**Goes in.** `## Competing interests`, L922.

### 2.4 CRediT author contributions — **blocking**

**Needed.** A CRediT statement per author across the relevant roles —
conceptualisation, methodology, software, formal analysis, investigation, data
curation, writing (original draft), writing (review and editing),
visualisation, supervision, project administration.

**Repository status.** **Partially evidenced, not determinable.** Git history
gives a per-area touch count:

| Author | scripts/ | src/ | tests/ | docs/ | manuscript/ | report/ |
|---|---:|---:|---:|---:|---:|---:|
| Mayank Tewatia | 76 | 37 | 37 | 44 | 27 | 12 |
| Parv | 31 | 31 | 10 | 4 | 26 | 32 |

**Why it matters.** Commit counts are not contributions. They do not capture
conceptualisation, supervision or review, they over-weight whoever committed
generated artifacts, and they say nothing about Dr. Sahasranand K. R., who has
no commits but appears as third author. **This table is a prompt for the
authors' memory, not a basis for assigning credit.** Its one firm use: it
confirms both first authors contributed substantially to software and analysis.

**Also needed.** Whether Dr. Sahasranand K. R.'s role is supervision, or
supervision plus other CRediT roles.

**Goes in.** `## Author contributions`, L926.

### 2.5 PPMI acknowledgement wording — **blocking**

**Needed.** PPMI's **current** required acknowledgement, verbatim, including
its sponsor and funding-partner list, plus the required data-citation format
and any version or download-date statement.

**Repository status.** **Absent and explicitly flagged by the authors
themselves** — `report/CARE_AI2026_Abstract.tex:109` carries
`% TODO: insert the current PPMI acknowledgement wording here.`

**Why it matters.** PPMI mandates specific wording and updates its partner
list. This is the one item where a plausible-looking paraphrase is actively
harmful: it would misstate a third party's required attribution and could
breach the Data Use Agreement. **No wording has been drafted here.** It must be
taken from PPMI's current guidance at submission, not from memory or from an
older paper.

**Goes in.** `## Acknowledgements`, L930, and the Data availability statement
at L900, which also awaits the DUA reference.

### 2.6 Repository hosting and availability — **blocking, but narrow**

**Needed.** The repository URL for the code-availability statement, and whether
it will be made public or deposited with a DOI (Zenodo or similar) at
acceptance. If it stays private, whether access is available to reviewers and
editors on request.

**Repository status.** The **code question is settled**: all analysis code is
tracked and complete, and the statement at L904 now says so. Only hosting is
open. Note `main` is currently **16 commits ahead of `origin/main`**, so the
remote does not yet carry this manuscript work — a release would need those
commits pushed.

**Why it matters.** The narrowest of the seven, and the only one that is a
logistics decision rather than a fact to be retrieved.

**Goes in.** `## Code availability`, L904, replacing the AUTHOR QUERY block.

### 2.7 ✅ RESOLVED 2026-11-04 — author list supplied by the authors

The authors supplied the list directly, superseding what the repository held:

> **Towards Longitudinal Prediction of Parkinson's Disease Motor Subtypes**
> Parv Patel¹, Mayank Tewatia¹, Varsha Vasudevan², and K. R. Sahasranand¹
> ¹ Indian Institute of Technology Palakkad, India · ² NeuroComp

Three things the repository had wrong or missing, now corrected throughout:

* **Varsha Vasudevan is a fourth author** and appears nowhere in the
  repository — not in any report, the CARE AI abstract, or the git history.
* **There are two affiliations**, not one. NeuroComp is new information.
* **K. R. Sahasranand is an author**, which settles the inconsistency recorded
  earlier between the CARE AI abstract (third author) and the project report
  (supervisor). The author form is used.

The title is also the authors' own, replacing the thesis-stating working
title.

**Still outstanding for this item:** the corresponding author, institutional
email addresses, and ORCID identifiers.

> **Note on the clean PDF.** `MANUSCRIPT_REVIEW_DRAFT.pdf` is now a clean
> reading copy at the authors' request: no review banner, and no author-query
> boxes. **Removing the boxes did not supply the declarations.** Ethics,
> funding, competing interests, CRediT and the PPMI acknowledgement remain
> absent, and §2.1–2.5 above are still the standing record of what is needed.
> The Data availability and Code availability statements are kept in the PDF
> because their prose is accurate on its own.

## 3. Journal requirements that could not be verified

Recorded as unverified rather than assumed. Stage 4 reached Nature's
content-types information through search, but **the live submission page sat
behind an authentication redirect and could not be fetched**, so the following
must be re-checked against the journal's own current guidance at submission.

| Requirement | What was found | Confidence |
|---|---|---|
| Abstract ≤ 150 words, unstructured | applied — the abstract is 147 | from a search result, **not** the live page |
| Title ≤ 15 words, no punctuation | applied — 14 words, punctuation-free | same |
| No strict word or page limit | assumed; the manuscript is ~7,980 words of prose | same |
| Figure, table and reference limits | **never verified** | unknown |
| AI-disclosure policy | **never verified** | unknown |
| Reporting-summary or checklist requirement | **never verified**; relevant to STROBE / TRIPOD+AI, which E-7 has not run | unknown |
| ORCID requirement | **never verified** | unknown |

**Recommended.** One pass over the journal's live author guidance before
submission, checking all seven rows. Two already shape the manuscript — the
abstract's word budget has three words of slack, and the title was rewritten to
satisfy a punctuation rule — so if either is wrong, both need revisiting.

---

## 4. One-line summary for the authors

Seven things are needed from you: **ethics basis, funding, competing interests
(each author), CRediT contributions, PPMI's current acknowledgement wording,
the repository hosting decision, and confirmation of author order,
affiliations, corresponding author and ORCIDs.** Six are facts only you hold.
The seventh, hosting, is a decision. None of them affects whether the science
is sound, and all of them block submission.
