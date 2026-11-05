#!/usr/bin/env python3
"""Re-check the audit deliverables against the repository's own result files.

Run from the repository root:

    ~/envs/ppmi/bin/python research_audit/verify_audit.py

Confirms that every document in `report/` and `reports/` is inventoried, that
every experiment id in the master table is named in the audit, that no metric
column was padded with a placeholder, and that a sample of numbers quoted in
the generated reports still matches the file it came from. Exits nonzero on
any failure.
"""
import pathlib, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUD = ROOT / "research_audit"
GEN = AUD / "generated_reports"
E = ROOT / "experiments"
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def norm(s):
    """Unicode minus and en-dash print differently from ASCII; compare on text."""
    return s.replace("−", "-").replace("–", "-")


inv = (AUD / "report_inventory.csv").read_text()
docs = sorted(p.name for p in ROOT.glob("report/*") if p.suffix in {".tex", ".md", ".html"})
docs += sorted(p.name for p in ROOT.glob("reports/*"))
missing = [d for d in docs if pathlib.Path(d).stem not in inv]
check(f"all {len(docs)} documents in report/ and reports/ are inventoried",
      not missing, ", ".join(missing) or "none missing")

mas = pd.read_csv(AUD / "master_experiment_results.csv")
audit = (AUD / "experiment_report_audit.md").read_text()
ids = sorted(mas.experiment_id.unique())
absent = [i for i in ids if i.split(" ")[0] not in audit]
check(f"all {len(ids)} experiment ids are named in the audit",
      not absent, ", ".join(absent) or "none absent")

metric_cols = ["balanced_accuracy", "precision", "auc", "brier_score", "log_loss", "log_likelihood"]
zeros = {c: int((mas[c].astype(str) == "0.0000").sum()) for c in metric_cols}
check("no metric column was padded with 0 in place of a missing value",
      all(v == 0 for v in zeros.values()), str(zeros))

e8 = pd.read_csv(E / "20261103T202632Z_e8_harmonised_family_comparison" / "harmonised_metrics.csv").set_index("model")
md = pd.read_csv(E / "20261103T202632Z_e8_harmonised_family_comparison" / "margin_decomposition.csv")
mr = md[(md.model == "CT-HMM, 3 latent states") & (md.current_label == "Indeterminate")].iloc[0]
e2 = pd.read_csv(E / "20261103T193205Z_e2_transition_patient_bootstrap" / "other_quantities.csv").set_index("quantity")
e2t = pd.read_csv(E / "20261103T193205Z_e2_transition_patient_bootstrap" / "transition_probabilities.csv")
ii = e2t[(e2t.from_state == "Indeterminate") & (e2t.to_state == "Indeterminate")].iloc[0]
o4 = pd.read_csv(E / "20261104T002308Z_o4_threshold_sensitivity" / "summary.csv").iloc[0]
ar = pd.read_csv(E / "20261104T002308Z_o4_threshold_sensitivity" / "attainable_ratios.csv").set_index("ratio")
e1 = pd.read_csv(E / "20261103T191408Z_e1_lag1_residual_resolution" / "lag1_statistic.csv").set_index("quantity")
a2p = pd.read_csv(E / "20261103T220147Z_a2p_outcome_table_fullprec" / "h3_component_vs_subtype.csv")
ct = pd.read_csv(E / "20261103T220147Z_a2p_outcome_table_fullprec" / "contrasts.csv").set_index("contrast")
fl = pd.read_csv(E / "20261103T192414Z_e5_participant_flow" / "flow_ledger.csv")

cases = [
    ("E-8 subscore GP indep accuracy", f"{e8.loc['Subscore GP, independent','accuracy']:.4f}"),
    ("E-8 CTMC accuracy",              f"{e8.loc['CTMC, interval-aware','accuracy']:.4f}"),
    ("E-8 persistence macro-F1",       f"{e8.loc['Persistence (recorded label)','macro_f1']:.4f}"),
    ("E-8 CT-HMM3 macro-F1",           f"{e8.loc['CT-HMM, 3 latent states','macro_f1']:.4f}"),
    ("E-8 Indeterminate share of gain", f"{100*mr.share_of_total_gain:.1f}"),
    ("E-8 Indeterminate share of cases", f"{100*mr.share_of_cases:.1f}"),
    ("E-8 Indeterminate cases gained",  str(int(mr.cases_gained))),
    ("E-2 reversion pooled",            f"{e2.loc['reversion_pooled','estimate']:.4f}"),
    ("E-2 KM naive 5y",                 f"{e2.loc['km_naive_yr5','estimate']:.4f}"),
    ("E-2 KM sustained 5y",             f"{e2.loc['km_sustained_yr5','estimate']:.4f}"),
    ("E-2 Indeterminate persistence",   f"{ii.probability:.4f}"),
    ("E-2 Indeterminate CI low",        f"{ii.bootstrap_low:.4f}"),
    ("E-5 pairs",                       f"{int(fl[fl.stage=='consecutive labelled visit pairs'].visits.iloc[0]):,}"),
    ("E-5 two-class eligible",          f"{int(fl[fl.stage.str.startswith('eligible for two-class')].visits.iloc[0]):,}"),
    ("O-4 visits relabelled",           str(int(o4.visits_relabelled))),
    ("O-4 share relabelled",            f"{100*o4.share_relabelled:.2f}"),
    ("O-4 patients affected",           str(int(o4.patients_with_any_relabelled))),
    ("O-4 share patients",              f"{100*o4.share_patients_affected:.2f}"),
    ("O-4 Indeterminate after",         str(int(o4.indeterminate_after))),
    ("O-4 visits at ratio 0.9091",      str(int(ar.loc[0.9091, 'n_visits']))),
    ("O-4 visits at ratio 1.1364",      str(int(ar.loc[1.1364, 'n_visits']))),
    ("E-1 observed lag-1",              f"{e1.loc['observed lag-1 correlation','estimate']:.4f}"),
    ("E-1 observed minus null",         f"{e1.loc['observed minus null','estimate']:.4f}"),
    ("A-2' MCI rate delta",             f"{a2p[(a2p.outcome=='mci')&(a2p.kind=='rate')].delta.iloc[0]:.4f}"),
    ("A-2' MCI level delta",            f"{a2p[(a2p.outcome=='mci')&(a2p.kind=='level')].delta.iloc[0]:.4f}"),
    ("A-2' profile vs LOCF",            f"{ct.loc['rich profile against assume-no-change','loglik_delta']:.4f}"),
    ("A-2' rate component",             f"{ct.loc['modelling the rate of change','loglik_delta']:.4f}"),
]

text = norm("\n".join(p.read_text() for p in GEN.glob("*.md")))
bad = [f"{label} ({value})" for label, value in cases if norm(value) not in text]
check(f"{len(cases)} numbers re-read from source and found in the generated reports",
      not bad, "; ".join(bad) or "all found")

for p in sorted(GEN.glob("*.md")):
    t = p.read_text()
    check(f"{p.name} carries Limitations / Reproducibility / Source evidence",
          all(h in t for h in ("## 8. Limitations", "## 9. Reproducibility", "## 11. Source evidence")))

print()
print("VERIFICATION:", "ALL PASS" if not fails else f"{len(fails)} FAILURE(S)")
sys.exit(1 if fails else 0)
