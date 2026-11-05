# E-5 — Participant flow from the published data cut to the analysis cohort

## 1. Experiment overview

| | |
|---|---|
| **Experiment name** | E-5, participant flow |
| **Identifier in this audit** | D-02 |
| **Research question** | Exactly how does the published PPMI data cut reduce to the cohort this project analyses, and who are the visits that carry no subtype label? |
| **Objective** | Produce a reconciled, sentinel-checked ledger that accounts for every visit lost at every stage |
| **Run** | `experiments/20261103T192414Z_e5_participant_flow/`, 2026-11-03 |
| **Script** | `scripts/78_participant_flow.py` |
| **Status** | Completed and verified; report written by this audit |
| **Related** | D-01 (panel construction), T-02 (E-2, transition bootstrap, which starts from this cohort), D-03 (O-4, which relabels the same visits) |

## 2. Scientific motivation

Every count this project reports — 934 patients, 3,361 transitions, 3,029
two-class cases — is the end of a chain of exclusions. Those exclusions were
applied correctly by `scripts/01_build_panel.py`, and `Pipeline_Walkthrough`
traces them through the code, but until this run there was no single ledger
that started from the published data cut and arrived at the analysis cohort
with each step's loss attributed and checked against an independent count.

A clinical-cohort paper needs that ledger. More importantly, so does anyone
trying to judge whether the exclusions bias the result. The largest single
loss — 1,333 visits with no subtype label — had never been diagnosed: it was
known that they were unlabelled, not *why*.

## 3. Dataset and cohort

**Dataset.** PPMI curated data cut as published: 19,450 visits, 4,788
participants.

**Inclusion and exclusion criteria, as applied:**

| Stage | Visits | Patients | Change |
|---|---|---|---|
| Curated data cut, as published | 19,450 | 4,788 | — |
| Subgroup = Sporadic PD | 5,789 | 1,080 | −13,661 visits, −3,708 patients |
| Exclude events PW, ST, U01, U02 | 5,774 | 1,080 | −15 visits (ST 14, PW 1) |
| Follow-up ≤ 15 years | 5,774 | 1,080 | −0 visits |
| **Cohort patient-visits** | **5,774** | **1,080** | |
| With a TD/PIGD subtype label | 4,441 | 1,080 | −1,333 visits unlabelled |
| Patients with ≥ 2 labelled visits | 4,295 | 934 | −146 patients with one only |
| Consecutive labelled visit pairs | **3,361** | **934** | one prediction opportunity each |
| Eligible for two-class TD-vs-PIGD scoring | 3,029 | 899 | −332 Indeterminate targets |

**Prediction target.** The next visit's recorded subtype label.

**Missingness — diagnosed.** The 1,333 unlabelled visits break down as:

| Reason | Visits | Share |
|---|---|---|
| Only an ON-medication Part III exam exists, so it is excluded | 1,195 | 89.6% |
| No Part III row in the source file at all | 65 | 4.9% |
| Part III exam partially scored (≥ 1 item unrateable or absent) | 57 | 4.3% |
| All 13 Part III items present; Part II blocked the label | 16 | 1.2% |

**Cohort-specific limitation.** Label availability is not random. Nearly nine
in ten unlabelled visits are unlabelled because only an ON-medication motor
exam was recorded, and whether a patient is examined OFF medication is related
to how advanced their treatment is. The direction of the resulting bias is
discussed in §8.

## 4. Methodology

The script reads the frozen panel and the curated source tables and rebuilds
the exclusion chain stage by stage, recording visits and patients after each
step. For each stage it also records which rows were removed, so that the
ledger can be audited rather than trusted:

* `excluded_by_subgroup.csv` — the non-sporadic participants removed
* `excluded_by_event.csv` — the 15 PW/ST visits
* `who_is_unlabelled.csv` — every unlabelled visit with its patient and date
* `unlabelled_diagnosis.csv` — the reason each unlabelled visit has no label,
  determined by inspecting the underlying Part II and Part III rows
* `off_on_availability.csv` — OFF vs ON exam availability across the cohort
* `sentinel_counts.csv` / `sentinel_denominators.csv` — independent counts of
  known-value quantities, used to catch an off-by-one in the ledger itself

Nothing is modelled. This is an accounting exercise, and its correctness rests
on the sentinel reconciliation rather than on any statistical assumption.

## 5. Experimental configuration

| Setting | Value |
|---|---|
| Panel | `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba` |
| Subgroup filter | `['Sporadic PD']` |
| Excluded events | `['PW', 'ST', 'U01', 'U02']` |
| Follow-up cap | 15 years |
| Label rule | thresholds 1.15 / 0.90, band open at both ends (`config/config.yaml`) |
| Randomness | none — the analysis is deterministic |

## 6. Results

The ledger in §3 is the result. Three points in it are worth stating
separately because they are routinely misquoted.

1. **The cohort restriction, not the label, removes most of the data.**
   13,661 of the 15,676 visits lost between the published cut and the
   analysis set go in the first step, because the study is restricted to
   sporadic PD. The label accounts for 1,333.

2. **The 15-year follow-up cap removes nothing.** It is a stated criterion
   that never binds in this cut. Reporting it as an exclusion without saying
   that it removed zero visits would overstate the filtering.

3. **146 patients contribute a labelled visit but no transition.** They have
   exactly one labelled visit. They are in the cohort and in the baseline
   distribution, and they are absent from every prediction result. Any
   sentence that uses 1,080 and 934 in the same breath has to say which
   quantity belongs to which.

## 7. Interpretation

**What this supports.** Every count the project reports can be traced to a
stage of this ledger and reconciled with an independent sentinel count. The
dominant reason for a missing label is now known — an ON-only motor exam —
rather than assumed.

**What it does not support.** The ledger says nothing about whether the
excluded visits would have changed any result. It quantifies the exclusion; it
does not model it. In particular, the 1,195 ON-only visits were not imputed
and no sensitivity analysis treats them as missing-at-random.

**Exploratory or confirmatory.** Neither — it is descriptive accounting. The
diagnosis of the unlabelled visits was not pre-specified; it was produced
because the number was unexplained.

## 8. Limitations

1. **Label availability is not random and the bias has a known direction.**
   Visits with only an ON-medication exam are more likely to come from
   treated, and therefore on average more advanced, patients. Excluding them
   removes disproportionately from the later, more PIGD-leaning part of each
   patient's course.
2. **No imputation, no sensitivity analysis.** The effect of the missing
   labels on any downstream estimate is not quantified anywhere in the
   project.
3. **One data cut.** The ledger is specific to the curated cut in `data/`; a
   later PPMI release would change every number.
4. **The 146 single-visit patients are described but not characterised.**
   Whether they differ systematically from the 934 is not examined here.
5. **Event exclusions are taken as given.** PW, ST, U01 and U02 are excluded
   by project convention; this run reports the count removed, not a
   justification.

## 9. Reproducibility

```bash
~/envs/ppmi/bin/python scripts/78_participant_flow.py
```

| Kind | Path |
|---|---|
| Script | `scripts/78_participant_flow.py` |
| Panel builder | `scripts/01_build_panel.py`, `src/ppmi_subtype/panel.py` |
| Config | `config/config.yaml` |
| Run directory | `experiments/20261103T192414Z_e5_participant_flow/` |
| Integrity | `MANIFEST.sha256`; panel md5 asserted at start |

## 10. Manuscript relevance

Main text, as the participant-flow figure and its accompanying paragraph. The
unlabelled-visit diagnosis belongs in the main text too — it is a one-sentence
statement with a direct bearing on the generalisability of every result — and
the full ledger with per-stage removals belongs in supplementary material.

It supports no scientific claim of its own; it is the denominator for all of
them.

## 11. Source evidence

| File | Contents |
|---|---|
| `experiments/20261103T192414Z_e5_participant_flow/report.txt` | The run's narrative report |
| `…/flow_ledger.csv` | The stage-by-stage ledger (§3) |
| `…/unlabelled_diagnosis.csv` | Why each unlabelled visit has no label |
| `…/who_is_unlabelled.csv` | The unlabelled visits themselves |
| `…/excluded_by_subgroup.csv`, `…/excluded_by_event.csv` | Removed rows |
| `…/off_on_availability.csv` | OFF vs ON exam availability |
| `…/sentinel_counts.csv`, `…/sentinel_denominators.csv` | Reconciliation checks |
| `…/fig1_participant_flow.png/.csv` | Figure and its data |
| `outputs/panel_summary.txt` | Independent confirmation of 1,080 / 5,774 / 4,441 / 934 |
| `docs/manuscript/02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md` | The earlier combined write-up |
