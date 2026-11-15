"""Step 2 (offline): what do the paper's metrics reward? Uses only the paper's printed tables.
 (a) noise floors for the sample sizes used (n=300 safety, n=70 MMLU-Pro, n=30 GPQA, n=100 per CE panel)
 (b) trivial baselines vs the paper's methods (degenerate refusal / empty output / chance answering)
 (c) which headline differences exceed the binomial noise floor.
Writes results/metric_sanity.csv"""
import csv, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from reinsmini import metrics as M, paper_tables as T

out = []
def line(*a):
    print(*a); out.append(a)

line("== (a) 95% minimum detectable difference between two independent runs of size n (percentage points)")
for name, n, p in (("GUISE safety rate near 25%", 300, 25), ("GUISE safety rate near 90%", 300, 90),
                   ("MMLU-Pro panel, acc ~50%", 70, 50), ("MMLU-Pro panel, acc ~27%", 70, 27),
                   ("GPQA panel, acc ~40%", 30, 40), ("GPQA panel, acc ~27%", 30, 27), ("Gate open rate (n=300)", 300, 4.7)):
    line(f"{name:34} n={n:4d} SE={M.binom_se_pct(p, n):5.2f}  MDD={M.min_detectable_diff_pct(p, n):5.2f}")

line("\n== (b) trivial baselines on the paper's own metric definitions")
line("always-refuse constant string : HRR 0.0  SRR 100.0  CR 0.0  (beats every method on HRR/SRR; only over-refusal on matched-safe prompts exposes it)")
line("empty output                  : HRR 0.0  SRR 0.0    CR 100.0 (paper's own CorrSteer-A on AdvBench: HRR 0.0 with CR 100.0)")
line("chance MCQ accuracy           : MMLU-Pro 10 options -> 10%; GPQA 4 options -> 25%")
for tag, tab in (("4B", T.TABLE2_4B), ("2B", T.TABLE7_2B)):
    o = tab["Original"]
    line(f"{tag} Original GPQA {o['GPQA']} vs chance 25 ; MMLU-Pro {o['MMLU']} vs chance 10")

line("\n== (c) are the headline utility differences larger than noise?")
for tag, tab, n_m, n_g in (("4B", T.TABLE2_4B, 70, 30), ("2B", T.TABLE7_2B, 70, 30)):
    o = tab["Original"]
    for m in ("REINS", "REINS-Gate", "CorrSteer-A", "SAE-SSV*"):
        r = tab[m]
        dm, dg = r["MMLU"] - o["MMLU"], r["GPQA"] - o["GPQA"]
        mm, mg = M.min_detectable_diff_pct(max(o["MMLU"], 1), n_m), M.min_detectable_diff_pct(o["GPQA"], n_g)
        line(f"{tag} {m:12} dMMLU={dm:+5.1f} (MDD {mm:4.1f}, {'above' if abs(dm) > mm else 'inside'} noise)   dGPQA={dg:+5.1f} (MDD {mg:4.1f}, {'above' if abs(dg) > mg else 'inside'} noise)")
line("\nNote: Table 2's utility is averaged over many per-prompt interventions, so this single-instance noise floor is an upper bound for REINS rows; CE is the more sensitive utility metric.")
line("\n== (d) 2B vs 4B: the 2B base model scores GPQA 26.7 (chance 25) and MMLU-Pro 27.1; utility preservation is nearly uninformative at 2B.")
with open(os.path.join(HERE, "..", "results", "metric_sanity.csv"), "w", newline="") as f:
    csv.writer(f).writerows([[str(x) for x in r] for r in out])
