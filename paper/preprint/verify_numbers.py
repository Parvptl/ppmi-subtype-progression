"""Re-derive the headline numbers quoted in the preprint from the committed
result files and report any mismatch.

Run from the repository root:  python -I paper/preprint/verify_numbers.py
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAILURES = []
CHECKS = 0


def check(label, got, want, tol=5e-4):
    global CHECKS
    CHECKS += 1
    if isinstance(want, bool) or isinstance(want, str) or isinstance(want, int):
        ok = got == want
    else:
        ok = abs(float(got) - float(want)) <= tol
    if not ok:
        FAILURES.append(f"{label}: paper says {want!r}, source gives {got!r}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<58} {got!r}")


def rows(rel):
    with open(ROOT / rel, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# -------------------------------------------------------- cohort and panel
print("\nCohort and frozen panel")
panel = rows("outputs/visit_panel.csv")
labelled = [r for r in panel if r["subtype_name"].strip()]
check("cohort visits", len(panel), 5774)
check("patients", len({r["PATNO"] for r in panel}), 1080)
check("labelled visits", len(labelled), 4441)

dist = Counter(r["subtype_name"] for r in labelled)
check("labelled TD", dist["TD"], 2891)
check("labelled Indeterminate", dist["Indeterminate"], 441)
check("labelled PIGD", dist["PIGD"], 1109)

per = defaultdict(int)
for r in labelled:
    per[r["PATNO"]] += 1
multi = {p for p, c in per.items() if c >= 2}
sub = [r for r in labelled if r["PATNO"] in multi]
check("patients with >=2 labelled visits", len(multi), 934)
check("labelled visits, modelling cohort", len(sub), 4295)

zero_all = [r for r in labelled if r["pigd_score"].strip() and float(r["pigd_score"]) == 0.0]
zero_sub = [r for r in sub if r["pigd_score"].strip() and float(r["pigd_score"]) == 0.0]
check("ratio undefined, full cohort", len(zero_all), 725)
check("  of which TD", Counter(r["subtype_name"] for r in zero_all)["TD"], 665)
check("  of which Indeterminate", Counter(r["subtype_name"] for r in zero_all)["Indeterminate"], 60)
check("ratio undefined, modelling cohort", len(zero_sub), 692)
check("ratio undefined, full cohort, %", round(100 * len(zero_all) / len(labelled), 1), 16.3)
check("ratio undefined, modelling cohort, %", round(100 * len(zero_sub) / len(sub), 1), 16.1)

# ------------------------------------------------- leave-one-patient-out
print("\nHeld-out three-class accuracy")
loo = {r["model"]: r for r in rows("outputs/loo/loo_model_comparison.csv")}
for name, acc, lo, hi in [
    ("HMM", 0.7539, 0.7348, 0.7731),
    ("Markov (discrete)", 0.7474, 0.7272, 0.7670),
    ("CTMC (interval-aware)", 0.7453, 0.7250, 0.7646),
    ("Persistence (stay)", 0.7227, 0.7009, 0.7437),
    ("Marginal (null)", 0.6293, 0.6011, 0.6565),
]:
    r = loo[name]
    check(f"{name} accuracy", float(r["accuracy"]), acc)
    check(f"{name} CI low", float(r["accuracy_ci_patient_bootstrap_low"]), lo)
    check(f"{name} CI high", float(r["accuracy_ci_patient_bootstrap_high"]), hi)
check("evaluation set size", int(loo["HMM"]["n_predictions"]), 3361)

pc = {r["state"]: r for r in rows("outputs/loo/loo_per_class_metrics.csv")}
check("HMM predicts Indeterminate n times", int(pc["Indeterminate"]["n_predicted"]), 1)
check("Indeterminate outcomes in evaluation set", int(pc["Indeterminate"]["support"]), 332)
check("HMM recall TD", round(float(pc["TD"]["recall"]), 4), 0.9102)
check("HMM recall PIGD", round(float(pc["PIGD"]["recall"]), 4), 0.6663)

# ------------------------------------------------------ growth model
print("\nHierarchical Bayesian growth model")
hb = {r["parameter"].split(" (")[0]: r for r in rows("outputs/hbayes/posterior_summary.csv")}
for key, mean, lo, hi in [
    ("beta0", 2.2920, 2.1577, 2.4272),
    ("beta1", -0.0927, -0.1159, -0.0682),
    ("sigma", 1.1380, 1.0993, 1.1778),
    ("tau0", 1.3408, 1.2466, 1.4421),
    ("tau1", 0.1033, 0.0780, 0.1315),
]:
    check(f"{key} posterior mean", round(float(hb[key]["mean"]), 4), mean)
    check(f"{key} CrI low", round(float(hb[key]["q2.5"]), 4), lo)
    check(f"{key} CrI high", round(float(hb[key]["q97.5"]), 4), hi)
check("sigma / band width", round(float(hb["sigma"]["mean"]) / 0.25, 1), 4.6)

# ------------------------------------------------------------ ratio GP
print("\nGaussian process on the ratio")
mc = {r["model"]: r for r in rows("outputs/step3_gpr/model_comparison.csv")
      if r["cohort"] == "unfiltered"}
check("ratio evaluation points", int(mc["K1"]["n"]), 2707)
for model, mae in [("K0", 1.005), ("K1", 0.972), ("K2", 0.972), ("K3", 0.971),
                   ("hbayes", 1.006), ("persistence", 1.043),
                   ("patient_mean", 1.063), ("population", 1.395)]:
    check(f"{model} MAE", round(float(mc[model]["MAE"]), 3), mae, tol=1e-9)
check("K1 coverage95", round(float(mc["K1"]["coverage95"]), 3), 0.941)
check("hbayes coverage95", round(float(mc["hbayes"]["coverage95"]), 3), 0.930)

pair = {(r["model_a"], r["model_b"]): r for r in
        rows("outputs/step3_gpr/paired_comparisons.csv") if r["cohort"] == "unfiltered"}
for a, b, d, lo, hi in [
    ("K1", "K0", -0.0333, -0.0467, -0.0205),
    ("K3", "K1", -0.0005, -0.0015, 0.0005),
    ("K0", "hbayes", -0.0009, -0.0033, 0.0014),
    ("K1", "persistence", -0.0707, -0.0979, -0.0451),
]:
    r = pair[(a, b)]
    check(f"{a} - {b} delta MAE", round(float(r["diff_pooled"]), 4), d)
    check(f"{a} - {b} CI low", round(float(r["pooled_lo"]), 4), lo)
    check(f"{a} - {b} CI high", round(float(r["pooled_hi"]), 4), hi)

# -------------------------------------------------------- subscore GP
print("\nGaussian processes on the subscores")
sp = {(r["outcome"], r["comparison"]): r for r in rows("outputs/joint_subscore_gp/paired.csv")}
for outcome, d, lo, hi, excl in [
    ("tremor", -0.0031, -0.0084, 0.0022, "False"),
    ("pigd", 0.0134, 0.0083, 0.0187, "True"),
]:
    r = sp[(outcome, "gp_indep_K1 - persistence")]
    check(f"{outcome}: GP - persistence", round(float(r["delta_MAE"]), 4), d)
    check(f"{outcome}: CI low", round(float(r["lo"]), 4), lo)
    check(f"{outcome}: CI high", round(float(r["hi"]), 4), hi)
    check(f"{outcome}: excludes zero", r["excludes_zero"], excl)

pz = {r["model"]: r for r in rows("outputs/joint_subscore_gp/pigd_zero_analysis.csv")}
check("visits with PIGD subscore zero", int(pz["persistence"]["n"]), 446)
check("persistence predicts <0.1 there", round(float(pz["persistence"]["frac_pigd_pred_below_0.1"]), 2), 0.62)
check("GP predicts <0.1 there", float(pz["gp_indep_K1"]["frac_pigd_pred_below_0.1"]), 0.0)
check("GP mean prediction there", round(float(pz["gp_indep_K1"]["pigd_pred_mean"]), 3), 0.242)

sm = {(r["model"], r["rule"]): r for r in rows("outputs/joint_subscore_gp/subtype_metrics.csv")}
check("GP-derived three-class accuracy", round(float(sm[("gp_indep_K1", "prob_subtype")]["accuracy"]), 4), 0.7566)

# ------------------------------------------------- matched GP vs HMM
print("\nMatched GP against HMM (recomputed from saved predictions)")
hmm = [r for r in rows("outputs/loo/loo_predictions.csv") if r["model"] == "HMM"]
gp = [r for r in rows("outputs/joint_subscore_gp/predictions_with_subtype.csv")
      if r["model"] == "gp_indep_K1"]
H, G = defaultdict(list), defaultdict(list)
for r in hmm:
    H[r["PATNO"]].append(r)
for r in gp:
    G[r["held_out_patient"]].append(r)
check("same patient set", set(H) == set(G), True)
check("matched patients", len(H), 934)

paired, agree = [], 0
for p in sorted(H):
    hs = sorted(H[p], key=lambda r: int(r["visit_index"]))
    gs = sorted(G[p], key=lambda r: int(r["target_index"]))
    for a, b in zip(hs, gs):
        agree += a["actual"] == b["obs_subtype"]
        paired.append((a["predicted"] == a["actual"], b["prob_subtype"] == a["actual"]))
check("matched prediction points", len(paired), 3361)
check("target labels agree everywhere", agree, 3361)
check("HMM accuracy", round(sum(a for a, _ in paired) / len(paired), 4), 0.7539)
check("GP-derived accuracy", round(sum(b for _, b in paired) / len(paired), 4), 0.7566)
check("paired difference", round((sum(b for _, b in paired) - sum(a for a, _ in paired)) / len(paired), 4), 0.0027)

# ---------------------------------------------------------------- report
print(f"\n{'=' * 72}")
if FAILURES:
    print(f"{len(FAILURES)} of {CHECKS} checks FAILED:\n")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print(f"All {CHECKS} checks passed.")
