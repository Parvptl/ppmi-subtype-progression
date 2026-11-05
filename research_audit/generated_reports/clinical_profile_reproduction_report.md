# A-2, A-2′ and A-3 — Reproducing the rich clinical profile, its saved-table precision, and the missing paired baseline

## 1. Experiment overview

| | |
|---|---|
| **Experiment names** | A-2, clinical-profile CV rerun at HEAD · A-2′, outcome-table round-trip verification · A-3, the same CV with the paired baseline |
| **Identifiers in this audit** | P-19, P-20, P-22 |
| **Research question** | The October rich-profile run recorded a commit at which its own module did not exist. Does it reproduce from committed code, and can its saved artifacts be reused? |
| **Objective** | Re-establish provenance for a run the manuscript depends on, and determine which of its saved files are safe to read |
| **Runs** | `experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/` and `experiments/20261103T220147Z_a2p_outcome_table_fullprec/`, both 2026-11-03; `experiments/20261104T085320Z_a3_profile_cv_with_baseline/`, 2026-11-04 |
| **Scripts** | `scripts/64_clinical_profile_cv.py` (the model), `scripts/83_compare_clinical_profile_runs.py` (A-2), `scripts/84_verify_outcome_table_roundtrip.py` (A-2′) |
| **Status** | Completed and verified; report written by this audit |
| **Related** | P-06 (the October run), P-21 (A-1, which depended on the lossy table), P-07, P-08 (which read P-06's outputs) |

## 2. Scientific motivation

Two defects were found in the rich clinical profile's October run during the
manuscript submission audit, and both bear on whether its results can be
cited.

**The provenance defect.** The run recorded commit `3a860ca`. At that commit
`src/ppmi_subtype/clinical_profile.py` **did not exist** — the module was
first committed five days later in `19bf8aa`. The run therefore executed
uncommitted code that was revised before it was committed, so nothing it saved
is reproducible from any committed state of the repository. That is a serious
claim to leave unresolved about a run supplying a main result.

**The precision defect.** `scripts/64` analysed an in-memory outcome table and
separately saved it with `.round(4)`. Four decimal places is destructive for
rate coordinates of order 10⁻³, so recomputing anything from the saved file
gave different answers than the run itself reported — a trap the submission
audit fell into and which produced the superseded A-1 numbers.

Neither question is about the science. Both are about whether the science can
be checked by anyone else.

## 3. Dataset and cohort

Identical in both the October run and the rerun:

| | |
|---|---|
| **Dataset** | Frozen visit panel, md5 `fb038ddbdc19ab3368e13940f398e7ba` |
| **Clinical visits** | 4,433 |
| **Patients with ≥ 2 visits** | 934 |
| **Prediction cases** | 3,353 |
| **Domains predicted** | 9 — `brady_rigid`, `tremor`, `pigd`, `updrs2`, `updrs1`, `moca`, `gds`, `scopa`, `ess` |
| **Medication** | LEDD is an **input**, not a predicted domain |
| **Representation** | each patient's filtered (level, rate) per domain — 18 continuous numbers with uncertainty attached, not a discrete state |
| **Missingness** | marginalised, not imputed |

## 4. Methodology

**A-2.** `scripts/64_clinical_profile_cv.py` was re-run at HEAD, with the A-1
orientation fix in place and the outcome table saved at full precision.
`scripts/83_compare_clinical_profile_runs.py` then diffs the two runs **file
by file and column by column**, reporting the largest shift in any column that
moves at all, against a tolerance of 5 × 10⁻⁵. Six files are named in the
script as the ones the manuscript quotes from, so that the manuscript's
exposure is explicit rather than assumed:

`representation_dimensionality.csv`, `outcome_associations.csv`,
`h3_component_vs_subtype.csv`, `contrasts.csv`,
`derived_label_vs_subtype_history.csv`, `model_comparison.csv`.

**A-2′.** `scripts/84_verify_outcome_table_roundtrip.py` recomputes the eight
outcome contrasts **from the saved `outcome_table.csv`** and checks them
against the run's own `h3_component_vs_subtype.csv`. If they agree, the saved
artifact is reusable; if not, it is still lossy.

## 5. Experimental configuration

| Setting | Value |
|---|---|
| Validation | 5-fold patient cross-validation, 4 repeats |
| Seed | 42 |
| Bootstrap draws (contrasts) | 2,000 |
| Comparison tolerance (A-2) | 5 × 10⁻⁵ |
| October run's recorded commit | `3a860ca` — predates `clinical_profile.py` |
| Module first committed in | `19bf8aa` |

## 6. Results

### 6.1 The rerun reproduces the October run exactly

Two-class derived-label results, October run against rerun at HEAD
(`derived_label_comparison.csv`), all on 3,021 scored cases:

| Model | n correct | Accuracy | Brier (2-class) | Log loss | Rerun |
|---|---|---|---|---|---|
| Profile: level only (random walk) | 2,550 | 0.8441 | 0.1195 | 0.3815 | **identical** |
| Profile: level + rate + medication | 2,539 | 0.8405 | 0.1177 | 0.3767 | **identical** |
| Profile: level + rate, no medication | 2,538 | 0.8401 | 0.1175 | 0.3760 | **identical** |
| Last observation carried forward | 2,492 | 0.8249 | 0.1235 | 0.3942 | **identical** |
| Population mean trajectory | 2,088 | 0.6912 | 0.2083 | 0.6069 | **identical** |

Every value matches to four decimal places. **The provenance defect did not
change any result.**

### 6.2 The saved outcome table now round-trips

A-2′ recomputes the eight contrasts from the saved full-precision
`outcome_table.csv` and reproduces the run's own
`h3_component_vs_subtype.csv`. The artifact is reusable.

The A-2 and A-2′ runs' `h3_component_vs_subtype.csv` files are byte-identical
to one another. Against the earlier lossy path the differences are:
MCI rate contrast 0.0448 against 0.0428 (−0.0020), and three others at
±0.0001. All four **level** contrasts are unchanged.

### 6.3 Headline profile results, from the verified rerun

Representation dimensionality (`representation_dimensionality.csv`):

| Coordinates | PC1 share | Cumulative, 2 PCs | Components for 90% | Effective rank |
|---|---|---|---|---|
| level | 0.4224 | 0.5587 | **7** | 4.34 |
| rate | 0.3723 | 0.4958 | **7** | 5.11 |

The profile is genuinely multidimensional — seven of nine components are
needed for 90% of the variance in both coordinate sets, so it is not a single
severity axis in disguise.

Profile-prediction contrasts, held-out log-density per domain observation
(`contrasts.csv`, n = 3,353 paired, 934 patients):

| Contrast | Δ log-density | 95% CI | Share of cases better |
|---|---|---|---|
| Rich profile vs assume-no-change | **+0.0733** | 0.0657 – 0.0813 | 75.1% |
| Rich profile vs the population average | +0.3889 | 0.3628 – 0.4167 | 86.6% |
| Modelling the rate of change | +0.0040 | 0.0019 – 0.0062 | 65.2% |
| Medication as an input | +0.0058 | 0.0041 – 0.0075 | 62.4% |

The rate component is worth +0.0040 of the model's +0.0733 advantage over
assume-no-change — about 5%.

Thirteen structural checks in `checks.json` pass, including *"every patient
appears in exactly one fold per repeat"*, *"each fold trained on the other
patients only"*, *"medication is not a predicted domain"* and *"no fold
failed"*.

### 6.4 The file that did not carry over, and why — A-3

The October run wrote `derived_label_vs_subtype_history.csv`, the table
holding the profile-versus-CT-HMM two-class contrast on 3,020 matched cases.
Neither the A-2 nor the A-2′ rerun directory contained it.

**The cause is a missing command-line argument, not a code change.**
`scripts/64_clinical_profile_cv.py:495` writes that table only inside
`if args.baseline:`, and `--baseline` defaults to `None`. The October run was
given the subtype-history run directory; neither rerun was. The reports
confirm it: the "Paired against the subtype-history experiment" section
appears once in the October `report.txt` and zero times in both reruns.

**A-3** re-ran the same script at commit `8268ea5` with the argument supplied,
through `scripts/cluster/submit.sh` so that provenance is recorded beside the
results. Elapsed 00:05:05 on `gpu03`, exit 0.

| | |
|---|---|
| Restored files | `derived_label_vs_subtype_history.csv`, `derived_label_contrasts.csv` |
| Agreement with the October versions | **maximum absolute difference 0.0** on every numeric column of both files |
| Six other shared outputs vs the A-2 rerun | **0.0** — so the 60-worker setting changes nothing |
| `scripts/64` in the recorded working diff | **no** |
| `src/ppmi_subtype/clinical_profile.py` in the recorded working diff | **no** |
| Provenance written | `git_commit.txt`, `git_diff.patch`, `untracked_source.tar.gz`, `pip_freeze.txt`, `job_header.txt` |

The restored table, as A-3 wrote it:

| Model | n scored | Accuracy | 95% CI | Brier (2-class) | Log loss |
|---|---|---|---|---|---|
| Profile: level only (random walk) | 3,020 | 0.8440 | 0.8285 – 0.8598 | 0.1195 | 0.3814 |
| Profile: level + rate + medication | 3,020 | 0.8404 | 0.8248 – 0.8566 | 0.1177 | 0.3766 |
| Profile: level + rate, no medication | 3,020 | 0.8404 | 0.8249 – 0.8563 | 0.1175 | 0.3759 |
| Subtype-history: CT-HMM (latent, interval-aware) | 3,020 | 0.8371 | 0.8202 – 0.8540 | 0.1215 | 0.3963 |
| Subtype-history: Persistence (last non-Indeterminate) | 3,020 | 0.8338 | 0.8167 – 0.8509 | 0.1598 | 4.1959 |
| Subtype-history: Markov (observed, discrete) | 3,020 | 0.8298 | 0.8130 – 0.8471 | 0.1327 | 0.4311 |
| Last observation carried forward | 3,020 | 0.8252 | 0.8083 – 0.8425 | 0.1235 | 0.3941 |
| Population mean trajectory | 3,020 | 0.6914 | 0.6677 – 0.7153 | 0.2083 | 0.6067 |

Persistence's log loss of 4.1959 is the clipped value for a degenerate 0/1
rule and is not comparable with the others.

`heldout_matched.csv` in the model-selection run (P-07) is a copy of the
October version and agrees with A-3's table row for row.

## 7. Interpretation

**What the results support.**

* The October rich-profile run is **numerically sound**. A rerun at HEAD,
  under committed code, reproduces every two-class figure exactly.
* The saved outcome table is now reusable: it round-trips to the run's own
  contrasts.
* The profile representation is multidimensional, and its advantage over an
  assume-no-change baseline is driven mainly by the level coordinates, with
  the rate component contributing about 5% of it.

**What the results do not support.**

* Reproducing the numbers does not retroactively make the October run
  reproducible. Anyone starting from commit `3a860ca` still cannot rebuild it.
  The correct citation for the profile results is the **rerun**, not the
  October run.
* A-3 restores the two-class contrast under committed code, but it does not
  show that the October run *would* have produced it — it shows that the
  committed code does, and that its output is identical.

**Exploratory or confirmatory.** Neither. These are verification runs. Their
value is in the provenance they establish, not in a hypothesis they test.

## 8. Limitations

1. ~~**The missing file.**~~ Resolved by A-3 (§6.4). What remains is that the
   paired comparison depends on a second run — the subtype-history experiment
   of 2026-10-20 — passed in by path, so the two-class contrast is only as
   reproducible as that run is.
2. **Exact reproduction was checked on the files the script names**, six of
   them. Files outside that list were compared but are not individually
   reported here.
3. **A-2′ verifies a round-trip, not correctness.** It shows the saved table
   and the in-memory analysis agree; both could share an error.
4. **The comparison tolerance is 5 × 10⁻⁵.** Differences below it are not
   reported.
5. **The profile's own limitations carry over** — optimistic component
   selection in the outcome contrasts, one cohort, one dataset. See
   `docs/latent_clinical_states/03_RICH_PROFILE_RESULTS_20261020.md`.

## 9. Reproducibility

```bash
~/envs/ppmi/bin/python scripts/64_clinical_profile_cv.py
~/envs/ppmi/bin/python scripts/83_compare_clinical_profile_runs.py \
    experiments/20261020T180515Z_clinical_profile_cv \
    experiments/20261103T213859Z_a2_clinical_profile_cv_rerun
~/envs/ppmi/bin/python scripts/84_verify_outcome_table_roundtrip.py \
    experiments/20261103T220147Z_a2p_outcome_table_fullprec
```

| Kind | Path |
|---|---|
| Model | `scripts/64_clinical_profile_cv.py`, `src/ppmi_subtype/clinical_profile.py` |
| Comparison tools | `scripts/83_compare_clinical_profile_runs.py`, `scripts/84_verify_outcome_table_roundtrip.py` |
| October run | `experiments/20261020T180515Z_clinical_profile_cv/` |
| Rerun | `experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/` |
| Full-precision rerun | `experiments/20261103T220147Z_a2p_outcome_table_fullprec/` |

Seed 42; 5-fold patient CV, 4 repeats; 2,000 bootstrap draws.

## 10. Manuscript relevance

**Supplementary, with one sentence in the data-availability or methods
statement.** The profile results belong in the main text; this report is the
provenance note that lets a reviewer verify them.

Two concrete obligations it creates:

1. Cite the **rerun** (`20261103T213859Z` / `20261103T220147Z`) as the source
   of every rich-profile number, not the October run.
2. ~~Either regenerate `derived_label_vs_subtype_history.csv` under committed
   code…~~ **Done.** Cite
   `experiments/20261104T085320Z_a3_profile_cv_with_baseline/derived_label_vs_subtype_history.csv`
   for the two-class contrast. It is byte-for-byte equivalent in value to the
   October table and was produced at commit `8268ea5` with the module
   unmodified in the working tree.

## 11. Source evidence

| File | Contents |
|---|---|
| `experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/report.txt` | The rerun's own report |
| `…/derived_label_comparison.csv` | §6.1 |
| `…/h3_component_vs_subtype.csv` | Authoritative outcome contrasts |
| `experiments/20261103T220147Z_a2p_outcome_table_fullprec/report.txt` | The full-precision run's report |
| `…/outcome_table.csv` | The saved table that now round-trips |
| `…/representation_dimensionality.csv`, `…/contrasts.csv` | §6.3 |
| `…/checks.json` | Thirteen structural checks |
| `experiments/20261104T085320Z_a3_profile_cv_with_baseline/derived_label_vs_subtype_history.csv` | **The two-class contrast, produced under committed code (A-3)** |
| `experiments/20261104T085320Z_a3_profile_cv_with_baseline/derived_label_contrasts.csv` | The paired contrasts against the CT-HMM, likewise restored |
| `experiments/20261104T085320Z_a3_profile_cv_with_baseline/job_header.txt`, `git_commit.txt`, `git_diff.patch`, `untracked_source.tar.gz`, `pip_freeze.txt` | A-3's provenance record |
| `experiments/20261020T180515Z_clinical_profile_cv/derived_label_vs_subtype_history.csv` | The October version; identical in value |
| `experiments/20261023T194317Z_model_selection_criteria/heldout_matched.csv` | A copy of the October version |
| `scripts/83_compare_clinical_profile_runs.py` | Provenance statement and the six manuscript-quoted files |
| `scripts/84_verify_outcome_table_roundtrip.py` | The round-trip check |
| `docs/manuscript/12_SUBMISSION_AUDIT_20261104.md` | Where findings A-2 and A-2′ were first recorded |
