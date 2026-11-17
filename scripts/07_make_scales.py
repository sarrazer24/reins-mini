"""Turn the fine scale scan into a per-layer input-scale file.
Rule (my choice, see REPRODUCTION.md 11): c_l = scale at which mean L0 on real activations equals the BatchTopK
target (128), log-interpolated, times --mult. NMSE-minimising scales are NOT used: NMSE keeps improving as the
scale grows (more features switch on), so its minimum does not identify the training scale."""
import argparse, collections, csv, json, math

def scales_from_scan(path, target=128.0, mult=1.0):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r["layer"])].append((float(r["scale"]), float(r["l0"])))
    out = {}
    for l, v in by.items():
        v.sort()
        for (c0, l0), (c1, l1) in zip(v, v[1:]):
            if l0 <= target <= l1 and l1 > l0:
                t = (target - l0) / (l1 - l0)
                out[str(l)] = round(math.exp(math.log(c0) + t * (math.log(c1) - math.log(c0))) * mult, 4)
                break
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="results/sae_scale_scan_fine.csv"); ap.add_argument("--target-l0", type=float, default=128.0)
    ap.add_argument("--mult", type=float, default=1.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = scales_from_scan(a.scan, a.target_l0, a.mult)
    json.dump(sc, open(a.out, "w"), indent=1); print(a.out, sc)
