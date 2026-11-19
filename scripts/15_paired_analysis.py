"""Paired per-prompt analysis of judged outputs (verdict files only, no responses).
Usage: python scripts/15_paired_analysis.py --runs runs/main --judge-id qwen25-7b --out results/paired_main.csv
Compares conditions on the SAME prompts: exact McNemar (sign test on discordant pairs) and a paired bootstrap CI of the rate difference."""
import argparse, collections, csv, glob, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--judge-id", required=True); ap.add_argument("--out", default="results/paired_main.csv")
a = ap.parse_args()

def load(sd, m, safe=False):
    f = os.path.join(sd, f"{m}{'_safe' if safe else ''}.judged_{a.judge_id}.jsonl")
    out = {}
    for l in open(f):
        r = json.loads(l); v = r["verdict"]
        out[r["sid"]] = None if v is None else M.assign_outcome(v["label"], v["refusal"], v["empty_or_degenerate"], v["semantic_coherent"])
    return out

def paired(x, y, outcome, rng, B=2000):
    ids = [k for k in x if k in y and x[k] and y[k]]
    ax = [x[k] == outcome for k in ids]; ay = [y[k] == outcome for k in ids]
    b = sum(p and not q for p, q in zip(ax, ay)); c = sum(q and not p for p, q in zip(ax, ay))
    n = len(ids); d = 100 * (sum(ay) - sum(ax)) / n
    bs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        bs.append(100 * sum(ay[i] - ax[i] for i in idx) / n)
    bs.sort()
    return dict(n=n, diff_pp=round(d, 1), ci_lo=round(bs[int(.025 * B)], 1), ci_hi=round(bs[int(.975 * B)], 1), only_first=b, only_second=c, p_mcnemar=round(M.paired_mcnemar_exact(b, c), 4))

PAIRS = [("original", "reins", "harmful"), ("original", "reins", "refusal"), ("refusal_only", "reins", "harmful"), ("refusal_only", "reins", "refusal"),
         ("original", "harm_only", "harmful"), ("random_active", "harm_only", "harmful"), ("original", "random_any", "harmful"),
         ("original", "refusal_only", "harmful"), ("reins", "reins_gate", "harmful")]
rows, rng = [], random.Random(0)
for sd in sorted(glob.glob(os.path.join(a.runs, "seed*"))):
    S = {m: load(sd, m) for m in ("original", "random_any", "random_active", "harm_only", "refusal_only", "reins")}
    g = {}
    gf = os.path.join(a.runs, "gate_scores_" + os.path.basename(sd) + ".csv")
    if os.path.isfile(gf):
        g = {r["sid"]: int(r["open"]) for r in csv.DictReader(open(gf)) if r["kind"] == "harmful"}
        S["reins_gate"] = {k: (S["reins"][k] if g.get(k, 0) else S["original"][k]) for k in S["reins"]}
    for x, y, o in PAIRS:
        if x in S and y in S:
            r = paired(S[x], S[y], o, rng); r.update(seed=os.path.basename(sd), first=x, second=y, outcome=o); rows.append(r)
            print(f"{os.path.basename(sd)} {o:8} {x:13} -> {y:13} diff {r['diff_pp']:+6.1f} pp [{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}]  discordant {r['only_first']}/{r['only_second']}  p={r['p_mcnemar']}")
    # per family
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in ("original", "refusal_only", "reins"):
        for k, v in S[m].items():
            fam[k.split("_")[0]][m].append(v == "harmful")
    for f_, d in sorted(fam.items()):
        print(f"{os.path.basename(sd)} family {f_:11} HRR original {100*sum(d['original'])/len(d['original']):5.1f}  refusal_only {100*sum(d['refusal_only'])/len(d['refusal_only']):5.1f}  reins {100*sum(d['reins'])/len(d['reins']):5.1f}  (n={len(d['reins'])})")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "first", "second", "outcome", "n", "diff_pp", "ci_lo", "ci_hi", "only_first", "only_second", "p_mcnemar"]); w.writeheader(); w.writerows(rows)
