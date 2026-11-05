## 4. Dataset

### 4.1 Source

All data come from the Parkinson's Progression Markers Initiative (PPMI) database [15], downloaded from the LONI Image and Data Archive on 20 August 2026 (the motor assessment files MDS-UPDRS Parts I to IV, the curated participant and visit tables, and the demographic, cognitive, autonomic, sleep, imaging and medication tables used as covariates). The curated data cut, as published, holds 19,450 visits of 4,788 participants across all PPMI cohorts. No PPMI data are reproduced in this paper or in its supplementary files; every table reports aggregates.

### 4.2 The subtype rule

The label is the Stebbins rule [1] re-implemented from the item level (Section 5). With the tremor score T the mean of the eleven tremor items and the PIGD score P the mean of the five gait and postural items, each item scored 0 to 4,

```
ratio r = T / P
TD             if r >= 1.15, or if P = 0 and T > 0
PIGD           if r <= 0.90
Indeterminate  if 0.90 < r < 1.15, or if P = 0 and T = 0
```

Both boundaries belong to the extreme classes (outer boundaries). Because T is a mean of eleven integers and P a mean of five, the ratio is a ratio of integer sums scaled by 5/11, and only a finite lattice of values is attainable; inside the Indeterminate band only eleven distinct ratio values occur in the cohort, and two of them, 10/11 = 0.9091 and 25/22 = 1.1364, hold 78% of the Indeterminate visits (Section 9.4). The re-implemented label agrees with PPMI's own derived label at 4,431 of 4,431 visits on the two-class TD/PIGD variable and at 1,595 of 1,597 visits (0.9987) on the three-class variable among the 390 participants for whom PPMI distributes it.

### 4.3 Cohort and participant flow

{{figure:1}}

{{table:1}}

The analysed cohort is the sporadic Parkinson's disease subgroup of the curated cut: 1,080 participants and 5,789 visits, reduced to 5,774 by excluding the premature-withdrawal, screening-termination and unscheduled events (15 visits). A visit carries a label only when all sixteen items are rated at an OFF-medication or untreated examination; 4,441 visits (76.9%) do. Table 2 accounts for the 1,333 visits that do not: at 1,195 of them (89.6%) only an ON-medication Part III examination exists, 65 have no Part III record, 57 have a partially scored examination, and the remainder lack the Part II items. Of the 1,080 participants, 934 have at least two labelled visits; they contribute 4,295 labelled visits and 3,361 consecutive labelled pairs, which are the prediction opportunities for every three-class model. Of those 3,361 pairs, 3,029 (899 participants) have a TD or PIGD outcome and are the eligible cases for every two-class model; the 332 pairs whose outcome is Indeterminate are counted but never scored on the two-class task.

{{table:2}}

### 4.4 Descriptive characteristics

{{table:3}}

Over all labelled visits the label is TD at 2,891 (65.1%), Indeterminate at 441 (9.9%) and PIGD at 1,109 (25.0%); at baseline it is TD at 772 (71.7%), Indeterminate at 109 (10.1%) and PIGD at 195 (18.1%) of 1,076 labelled baseline visits, matching the PPMI baseline distribution reported by Simuni and colleagues [4]. The ratio is undefined at 725 labelled visits (16.3%) because the PIGD score is zero; 665 of those are TD by the degenerate-case rule and 60 are Indeterminate. The median interval between consecutive labelled visits is 1.00 years (10th to 90th percentile 0.91 to 2.00), and the median labelled follow-up per participant is 3.1 years (10th to 90th percentile 1.2 to 11.0). The share of labelled visits at which the ratio is defined rises from 0.74 at baseline to above 0.90 after year five (Table 32), so the ratio-based models see a cohort that drifts towards the PIGD-bearing participants.

{{figure:7}}

{{figure:10}}

### 4.5 The frozen panel

Every experiment reads one frozen visit panel, `outputs/visit_panel.csv`, whose MD5 checksum (fb038ddbdc19ab3368e13940f398e7ba) is verified by each run script before fitting. The panel was built once by the pipeline described in Section 5 and never rebuilt; the threshold-sensitivity experiment (Section 9.3) wrote a separate perturbed panel with its own checksum and left the frozen panel untouched. The project's outputs directory is read-only by rule, and every later experiment writes to its own timestamped directory under `experiments/`.
