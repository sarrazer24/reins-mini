"""Step 0 (offline, no model): are the paper's tables internally consistent?
Checks row sums, granularity (k/300 safety; k/70 MMLU-Pro and k/30 GPQA for single-instance rows)
and the relative claims in the text. Writes results/paper_table_checks.csv.
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from reinsmini import paper_tables as P

rows = []
def chk(name, ok, detail):
    rows.append(dict(check=name, status="PASS" if ok else "FAIL", detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

def on_grid(x, n, tol=0.06):
    return abs(round(x * n / 100) * 100 / n - x) <= tol

for tag, tab in (("4B", P.TABLE2_4B), ("2B", P.TABLE7_2B)):
    for m, r in tab.items():
        s = r["HRR"] + r["SRR"] + r["OSR"] + r["CR"]
        chk(f"{tag} rowsum {m}", abs(s - 100) <= 0.15, f"sum={s:.1f}")
        vals = [r[k] for k in ("HRR", "SRR", "OSR", "CR")]
        n_ok = next((n for n in (300, 600, 900) if all(on_grid(v, n, 0.051) for v in vals)), None)
        # INFO, not FAIL: a cell that is off the k/300 grid is probably an average over several runs
        # (paper App. D.6 mentions repeat seeds) - this is my inference, the paper does not say.
        rows.append(dict(check=f"{tag} grid {m}", status="INFO",
                         detail=f"smallest consistent n in (300,600,900): {n_ok}"))
        print(f"[INFO] {tag} grid {m}: smallest consistent n in (300,600,900) = {n_ok}")
    o = tab["Original"]
    chk(f"{tag} Original MMLU-Pro on k/70 grid", on_grid(o["MMLU"], P.N_MMLU_PRO), f"{o['MMLU']} ~ {round(o['MMLU']*.7)}/70")
    chk(f"{tag} Original GPQA on k/30 grid", on_grid(o["GPQA"], P.N_GPQA), f"{o['GPQA']} ~ {round(o['GPQA']*.3)}/30")

for tag, tab in (("4B", P.TABLE2_4B), ("2B", P.TABLE7_2B)):
    c, r = tab["CorrSteer-A"], tab["REINS"]
    red = 100 * (c["HRR"] - r["HRR"]) / c["HRR"]
    ratio = r["SRR"] / c["SRR"]
    pr = P.TEXT_CLAIMS[f"{tag}_HRR_reduction_vs_CorrSteerA_pct"]; pq = P.TEXT_CLAIMS[f"{tag}_SRR_ratio_vs_CorrSteerA"]
    chk(f"{tag} HRR reduction vs CorrSteer-A", abs(red - pr) < 0.06, f"recomputed {red:.2f}% (paper {pr})")
    chk(f"{tag} SRR ratio vs CorrSteer-A", abs(ratio - pq) < 0.06, f"recomputed {ratio:.2f}x (paper {pq})")

for j, r in P.TABLE6_JUDGES.items():
    s = sum(r.values()); chk(f"Table6 rowsum {j}", abs(s - 100) <= 0.05, f"sum={s:.2f}")
for m, r in P.TABLE10_OVERREFUSAL_4B.items():
    s = sum(r.values()); chk(f"Table10 rowsum {m}", abs(s - 100) <= 0.05, f"sum={s:.2f}")

out = os.path.join(HERE, "..", "results"); os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "paper_table_checks.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["check", "status", "detail"]); w.writeheader(); w.writerows(rows)
bad = [r for r in rows if r["status"] == "FAIL"]
print(f"\n{len(rows)-len(bad)}/{len(rows)} checks passed")
for r in bad: print("FAILED:", r["check"], r["detail"])
