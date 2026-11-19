"""Stacked outcome bars per method from results/summary_main.csv (mean over seeds, error bar = seed range on HRR)."""
import csv, os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open("results/summary_main.csv")))
by = collections.defaultdict(list)
for r in rows: by[r["method"]].append(r)
order = [m for m in ("original", "random_any", "random_active", "harm_only", "refusal_only", "reins", "reins_gate") if m in by]
cols = dict(HRR="#c0392b", SRR="#2980b9", OSR="#27ae60", CR="#8e44ad")
fig, ax = plt.subplots(figsize=(9, 4))
bottom = [0] * len(order)
for k in ("HRR", "SRR", "OSR", "CR"):
    vals = [sum(float(r[k]) for r in by[m]) / len(by[m]) for m in order]
    ax.bar(order, vals, bottom=bottom, label=k, color=cols[k]); bottom = [b + v for b, v in zip(bottom, vals)]
for i, m in enumerate(order):
    h = [float(r["HRR"]) for r in by[m]]
    ax.plot([i, i], [min(h), max(h)], color="black", lw=2)
ax.set_ylabel("% of valid judged outputs"); ax.set_title("GUISE eval outcomes (black line: HRR range across seeds)"); ax.legend(ncol=4)
plt.xticks(rotation=20); plt.tight_layout(); os.makedirs("results", exist_ok=True); plt.savefig("results/outcomes.png", dpi=150)
print("wrote results/outcomes.png")
