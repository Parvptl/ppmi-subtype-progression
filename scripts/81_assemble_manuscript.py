#!/usr/bin/env python3
"""Assemble the full manuscript from the stage drafts (Stage 11).

The stage drafts in docs/manuscript/ follow one convention: manuscript prose
lives in `## `/`### ` sections before the first `# §` appendix heading, and
every editorial note is a `>` blockquote. This script extracts the prose,
applies the final citation numbering, and emits one continuous manuscript.

It is deliberately mechanical. Nothing here rewrites prose; if the assembled
text is wrong, the stage draft is wrong and should be fixed there and this
re-run.

Citation numbering is order of first appearance across the assembled
manuscript. All twelve references are first cited in the Introduction, so the
Introduction's numbering carries unchanged; Results and Methods cite by name
and have their markers inserted here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "manuscript"
OUT = ROOT / "manuscript" / "npj" / "MANUSCRIPT_FULL_20261104.md"

DRAFTS = {
    "introduction": DOCS / "09_INTRODUCTION_DRAFT_20261104.md",
    "results": DOCS / "08_RESULTS_DRAFT_20261104.md",
    "discussion": DOCS / "10_DISCUSSION_DRAFT_20261104.md",
    "methods": DOCS / "07_METHODS_DRAFT_20261104.md",
    "abstract": DOCS / "11_ABSTRACT_TITLE_DRAFT_20261104.md",
}

TITLE = ("Label reliability rather than model capacity limits longitudinal "
         "prediction of Parkinson's disease motor subtypes")

SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup(n: int) -> str:
    return str(n).translate(SUPERSCRIPT)


# Markers to insert where a draft cites by author name rather than by number.
# (file key, exact text in the draft, reference number)
NAME_CITATIONS = [
    ("results", "by Simuni and colleagues, who reported", 4),
    ("methods", "Markov model with classification error of Jackson et al., applied", 11),
]

# Final reference list, in order of first appearance. Strings are copied from
# the Stage 3 verified list; only the ordering differs.
REFERENCES = [
    "**Stebbins GT, Goetz CG, Burn DJ, Jankovic J, Khoo TK, Tilley BC.** How to identify tremor dominant and postural instability/gait difficulty groups with the Movement Disorder Society Unified Parkinson's Disease Rating Scale: comparison with the Unified Parkinson's Disease Rating Scale. *Mov Disord* 2013;28(5):668–670. doi:10.1002/mds.25383",
    "**Mestre TA, Fereshtehnejad SM, Berg D, et al.** Parkinson's disease subtypes: critical appraisal and recommendations. *J Parkinsons Dis* 2021;11(2):395–404. doi:10.3233/JPD-202472",
    "**Kotagal V.** Is PIGD a legitimate motor subtype in Parkinson disease? *Ann Clin Transl Neurol* 2016;3(6):473–477. doi:10.1002/acn3.312",
    "**Simuni T, Caspell-Garcia C, Coffey C, Lasch S, Tanner C, Marek K.** How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? *Parkinsonism Relat Disord* 2016;28:62–67. doi:10.1016/j.parkreldis.2016.04.027. PMID:27132498",
    "**Lee JW, Song YS, Kim H, Ku BD, Lee WW.** Alteration of tremor dominant and postural instability gait difficulty subtypes during the progression of Parkinson's disease: analysis of the PPMI cohort. *Front Neurol* 2019;10:471. doi:10.3389/fneur.2019.00471. PMID:31133973",
    "**Erro R, Picillo M, Amboni M, et al.** Comparing postural instability and gait disorder and akinetic-rigid subtyping of Parkinson disease and their stability over time. *Eur J Neurol* 2019;26:1212–1218. doi:10.1111/ene.13968",
    "**von Coelln R, Gruber-Baldini AL, Reich SG, Armstrong MJ, Savitt JM, Shulman LM.** The inconsistency and instability of Parkinson's disease motor subtypes. *Parkinsonism Relat Disord* 2021;88:13–18. doi:10.1016/j.parkreldis.2021.05.016. PMID:34091412",
    "**Ren J, Pan C, Li Y, Li L, Hua P, Xu L, Zhang L, Zhang W, Xu P, Liu W.** Consistency and stability of motor subtype classifications in patients with de novo Parkinson's disease. *Front Neurosci* 2021;15:637896. doi:10.3389/fnins.2021.637896. PMID:33732106",
    "**Severson KA, Chahine LM, Smolensky LA, Dhuliawala M, Frasier M, Ng K, Ghosh S, Hu J.** Discovery of Parkinson's disease states and disease progression modelling: a longitudinal data study using machine learning. *Lancet Digit Health* 2021;3(9):e555–e564. doi:10.1016/S2589-7500(21)00101-1. PMID:34334334",
    "**Dadu A, Satone V, Kaur R, et al.** Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. *npj Parkinsons Dis* 2022;8(1):172. doi:10.1038/s41531-022-00439-z. PMID:36526647",
    "**Jackson CH, Sharples LD, Thompson SG, Duffy SW, Couto E.** Multistate Markov models for disease progression with classification error. *J R Stat Soc Ser D (The Statistician)* 2003;52(2):193–209. doi:10.1111/1467-9884.00351",
    "**Bouwmeester W, Zuithoff NPA, Mallett S, Geerlings MI, Vergouwe Y, Steyerberg EW, Altman DG, Moons KGM.** Reporting and methods in clinical prediction research: a systematic review. *PLoS Med* 2012;9(5):e1001221. doi:10.1371/journal.pmed.1001221. PMID:22629234",
]

FIGURE_LEGENDS = [
    ("Fig. 1 | Participant flow.",
     "Of 4,788 participants and 19,450 visits in the curated data cut, 1,080 "
     "participants contributing 5,774 visits met the cohort criteria. A motor "
     "subtype could be assigned at 4,441 of those visits (76.9%), and 934 "
     "participants had at least two labelled visits, contributing 3,361 "
     "consecutive labelled transitions. Each exclusion is counted at the step "
     "that applied it."),
    ("Fig. 2 | Instability of the recorded label.",
     "**a**, Transition probabilities between consecutive labelled visits, with "
     "95% patient-clustered bootstrap intervals (2,000 resamples of "
     "participants). **b**, Proportion of observed changes of state that "
     "returned to the origin state at the following visit, by change type. "
     "**c**, Transition probabilities stratified by the elapsed interval "
     "between the two visits."),
    ("Fig. 3 | Harmonised comparison across five model families.",
     "Accuracy of nine fitted specifications and five reference rules on the "
     "same 3,361 transitions, re-scored onto one decision rule, one evaluation "
     "set and one baseline. Every predicted class is the argmax of a "
     "three-label probability vector; subscore models are propagated through "
     "the unmodified classification rule by Monte Carlo. Label persistence is "
     "the canonical reference."),
    ("Fig. 4 | Where the margin over persistence comes from.",
     "Each model's gain and loss against label persistence, decomposed by the "
     "label recorded at the prediction point. Counts sum to each model's total "
     "margin and cover all 3,361 transitions."),
    ("Fig. 5 | The continuous multi-domain representation.",
     "**a**, Share of across-participant variance explained by successive "
     "principal components of the level and rate coordinates; seven components "
     "are required for 90% of either. **b**, Discrimination of subsequently "
     "recorded outcomes, never used in fitting, by the best single coordinate "
     "against the subtype label. Coordinates were selected on the same data "
     "against which they are compared, so each difference is an upper bound."),
]

TABLE_CAPTIONS = [
    ("Table 1 | Cohort characteristics.",
     "Full eligible cohort (1,080 participants) and the analysed subset that "
     "contributed at least one prediction opportunity (934 participants)."),
    ("Table 2 | Transition probabilities between consecutive labelled visits.",
     "Nine cells with 95% patient-clustered bootstrap intervals, and the "
     "design effect against an interval that treats transitions as "
     "independent."),
    ("Table 3 | Harmonised three-class comparison.",
     "All fourteen specifications on the same 3,361 transitions: accuracy, "
     "multiclass Brier score, log loss, per-class recall, the number of "
     "predictions issued for each class, and macro-F1."),
    ("Table 4 | Two-class secondary analysis.",
     "Tremor dominant against PIGD on the 3,029 transitions whose next "
     "recorded label was one of the two, using conditional probabilities. The "
     "332 transitions with an indeterminate outcome are reported separately "
     "and never scored."),
    ("Table 5 | Per-participant accuracy by recorded-label composition.",
     "Exploratory. Groupings were defined after the predictions were generated "
     "and from recorded labels only; the persistence rule is shown on the same "
     "groups."),
]

ASSETS = [
    ("Fig. 1", "experiments/20261103T192414Z_e5_participant_flow/fig1_participant_flow.png", "run figure"),
    ("Fig. 2", "manuscript/npj/fig2_label_instability.png", "submission-ready"),
    ("Fig. 3", "experiments/20261103T202632Z_e8_harmonised_family_comparison/fig1_harmonised_family_comparison.png", "run figure"),
    ("Fig. 4", "manuscript/npj/fig4_margin_decomposition.png", "submission-ready"),
    ("Fig. 5a", "experiments/20261020T180515Z_clinical_profile_cv/representation_dimensionality.csv", "NOT DRAWN — source data only"),
    ("Fig. 5b", "manuscript/npj/fig5b_outcome_discrimination.png", "submission-ready"),
    ("Table 1", "manuscript/npj/table1_cohort_characteristics.csv", "submission-ready"),
    ("Table 2", "experiments/20261103T193205Z_e2_transition_patient_bootstrap/", "source CSVs, not formatted"),
    ("Table 3", "experiments/20261103T202632Z_e8_harmonised_family_comparison/harmonised_metrics.csv", "source CSV, not formatted"),
    ("Table 4", "outputs/joint_subscore/subtype_metrics.csv + auc_metrics.csv", "source CSVs, not formatted"),
    ("Table 5", "experiments/20261103T175003Z_patient_level_accuracy_audit/patient_accuracy_by_subtype_composition.csv", "source CSV, not formatted"),
]


def extract_prose(path: Path, demote: bool = False) -> str:
    """Prose from the first `## ` heading to the first `# §` appendix.

    `demote` lowers every heading one level, for drafts whose subsections are
    `## ` but which sit under a section heading this script adds (Results,
    Methods).
    """
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## "))
    try:
        end = next(i for i, l in enumerate(lines) if re.match(r"^# §", l))
    except StopIteration:
        end = len(lines)

    kept = []
    for line in lines[start:end]:
        if line.lstrip().startswith(">"):
            continue          # editorial note
        if line.strip() == "---":
            continue          # section rule
        # strip my working enumeration from headings: "## M1. X" -> "## X"
        line = re.sub(r"^(#{2,3}) (?:[MR]\d+[a-z]?)\. ", r"\1 ", line)
        if demote and line.startswith("#"):
            line = "#" + line
        kept.append(line)

    text = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_abstract(path: Path) -> str:
    body = path.read_text().split("## Abstract", 1)[1]
    body = body.split("**147 words**", 1)[0]
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith(">"))
    return re.sub(r"\n{2,}", " ", body).strip()


def main() -> int:
    parts: dict[str, str] = {}
    for key, path in DRAFTS.items():
        if not path.exists():
            print(f"MISSING DRAFT: {path}", file=sys.stderr)
            return 1
        if key == "abstract":
            parts[key] = extract_abstract(path)
        else:
            parts[key] = extract_prose(path, demote=key in ("results", "methods"))

    # Insert author-name citation markers.
    inserted = 0
    for key, needle, num in NAME_CITATIONS:
        flat = " ".join(parts[key].split())
        if " ".join(needle.split()) not in flat:
            print(f"CITATION ANCHOR NOT FOUND in {key}: {needle!r}", file=sys.stderr)
            return 1
        # Operate on the wrapped text: match the needle across line breaks.
        pattern = r"\s+".join(map(re.escape, needle.split()))
        # Place the marker after the cited name, before the following words.
        if key == "results":
            parts[key], n = re.subn(
                r"(by\s+Simuni\s+and\s+colleagues)", r"\1" + sup(num), parts[key], count=1)
        else:
            parts[key], n = re.subn(
                r"(of\s+Jackson\s+et\s+al\.)", r"\1" + sup(num), parts[key], count=1)
        if n != 1:
            print(f"MARKER INSERT FAILED in {key}", file=sys.stderr)
            return 1
        inserted += n

    doc: list[str] = []
    doc.append(f"# {TITLE}")
    doc.append("")
    doc.append("> **Assembled by `scripts/81_assemble_manuscript.py` on "
               "2026-11-04 from the Stage 6–10 drafts.** Manuscript text "
               "only: every editorial note in those drafts is excluded here and "
               "remains in the draft it belongs to. Regenerate rather than "
               "edit this file.")
    doc.append("")
    doc.append("**Authors.** Parv Patel, Mayank Tewatia, Sahasranand K. R. "
               "*(order and affiliations to be confirmed by the authors; see "
               "Author contributions below.)*")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Abstract")
    doc.append("")
    doc.append(parts["abstract"])
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append(parts["introduction"])
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Results")
    doc.append("")
    doc.append(parts["results"])
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append(parts["discussion"])
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Methods")
    doc.append("")
    doc.append(parts["methods"])
    doc.append("")
    doc.append("---")
    doc.append("")

    doc.append("## Data availability")
    doc.append("")
    doc.append("Data were obtained from the Parkinson's Progression Markers "
               "Initiative (PPMI) database and are available to qualified "
               "researchers from the PPMI repository at the Laboratory of "
               "Neuro Imaging (LONI) under PPMI's Data Use Agreement. The "
               "authors are not permitted to redistribute participant-level "
               "data. The analysis operates on a frozen visit panel whose "
               "checksum is recorded in the repository so that a reader with "
               "PPMI access can reconstruct the identical panel.")
    doc.append("")
    doc.append("> **TO BE COMPLETED BEFORE SUBMISSION.** The current PPMI "
               "acknowledgement wording and the DUA reference are not in the "
               "repository (Stage 1 C5); `report/CARE_AI2026_Abstract.tex` "
               "carries a `% TODO` for exactly this.")
    doc.append("")
    doc.append("## Code availability")
    doc.append("")
    doc.append("All analysis code is version-controlled: the data pipeline, "
               "the observed-label Markov chains, the continuous-time hidden "
               "Markov model, the subscore models and the Gaussian-process "
               "implementation on which the subscore Gaussian process in the "
               "primary comparison depends, the continuous multi-domain "
               "representation, every analysis script, and an accompanying "
               "test suite of 758 tests. No part of the analysis depends on "
               "code outside the repository.")
    doc.append("")
    doc.append("Derived outputs are deliberately not included. They contain "
               "participant-level values and are not redistributable under "
               "PPMI's Data Use Agreement. A reader with PPMI access can "
               "regenerate them by running the pipeline against the frozen "
               "visit panel, whose checksum the repository records.")
    doc.append("")
    doc.append("> **AUTHOR QUERY.** The repository URL, and whether it will be "
               "made public or deposited with a DOI at acceptance, must be "
               "supplied by the authors. The code itself is complete and "
               "version-controlled; only the hosting decision is outstanding.")
    doc.append("")
    for heading, note in [
        ("Ethics declarations",
         "**ABSENT (Stage 1 C5).** No ethics approval, consent statement or "
         "institutional position on secondary analysis of PPMI exists anywhere "
         "in the repository. Required before submission."),
        ("Funding",
         "**ABSENT (Stage 1 C5).** Required before submission."),
        ("Competing interests",
         "**ABSENT (Stage 1 C5).** Required before submission."),
        ("Author contributions",
         "**ABSENT (Stage 1 C5).** A CRediT statement must be supplied by the "
         "authors. Author list above is taken from the CARE AI 2026 abstract "
         "and is unconfirmed."),
        ("Acknowledgements",
         "**ABSENT (Stage 1 C5).** PPMI requires specific acknowledgement "
         "wording, including its sponsor and funding-partner list. Not in the "
         "repository."),
    ]:
        doc.append(f"## {heading}")
        doc.append("")
        doc.append("> " + note)
        doc.append("")

    doc.append("## References")
    doc.append("")
    for i, ref in enumerate(REFERENCES, 1):
        doc.append(f"{i}. {ref}")
    doc.append("")
    doc.append("> **Not cited, deliberately.** The Stage 3 verified list also "
               "contains Collins 2024 (TRIPOD+AI) and von Elm 2007 (STROBE). "
               "Neither is cited because the manuscript has **not** been "
               "checked against either checklist (Stage 5 E-7 was never run). "
               "Citing a reporting standard the manuscript has not been "
               "assessed against would assert a compliance that does not "
               "exist. Run E-7, then add both.")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Figure legends")
    doc.append("")
    for head, body in FIGURE_LEGENDS:
        doc.append(f"**{head}** {body}")
        doc.append("")
    doc.append("## Table captions")
    doc.append("")
    for head, body in TABLE_CAPTIONS:
        doc.append(f"**{head}** {body}")
        doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Asset status")
    doc.append("")
    doc.append("Where each display item currently stands. "
               "\"Run figure\" means a figure exists from its analysis run but "
               "has not been redrawn to journal specification.")
    doc.append("")
    doc.append("| Item | Source | Status |")
    doc.append("|---|---|---|")
    for item, src, status in ASSETS:
        doc.append(f"| {item} | `{src}` | {status} |")
    doc.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(doc) + "\n")

    words = len(" ".join(
        parts[k] for k in ("abstract", "introduction", "results", "discussion", "methods")
    ).split())
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  citation markers inserted: {inserted}")
    print(f"  references: {len(REFERENCES)}")
    print(f"  manuscript prose: ~{words} words "
          f"(abstract+intro+results+discussion+methods)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
