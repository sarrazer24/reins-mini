"""Inspect the real GUISE files: counts, schema, id uniqueness, my split; optional cross-check against the
authors' own split function (their clone is only imported for this comparison, never copied)."""
import argparse, collections, importlib.util, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--guise-dir", required=True)
ap.add_argument("--authors-clone", default=None, help="path to a clone of Geralt1020/REINS (optional cross-check)")
ap.add_argument("--seed", type=int, default=12)
a = ap.parse_args()

S = D.load_guise(a.guise_dir)
print("samples:", len(S), "| unique ids:", len({s.sid for s in S}))
print("categories:", dict(collections.Counter(s.category for s in S)))
print("missing matched-safe prompt:", sum(s.safe is None for s in S))
cal, ev = D.split_2to1(S, seed=a.seed)
print(f"my split seed={a.seed}: calibration={len(cal)} evaluation={len(ev)}",
      dict(collections.Counter(s.category for s in ev)))
os.makedirs("results", exist_ok=True)
json.dump({"seed": a.seed, "evaluation_ids": sorted(s.sid for s in ev)}, open(f"results/eval_split_seed{a.seed}.json", "w"))
print("wrote results/eval_split_seed%d.json (ids only, no text)" % a.seed)
if a.authors_clone:
    p = os.path.join(a.authors_clone, "src", "data.py")
    spec = importlib.util.spec_from_file_location("authors_data", p); m = importlib.util.module_from_spec(spec)
    sys.modules["authors_data"] = m; spec.loader.exec_module(m)
    rows = m.load_samples({"adapter": "guise"}, config_path=os.path.join(a.authors_clone, "configs", "x.json"))
    c2, e2 = m.split_samples(rows, {"mode": "stratified_shuffle_v1", "calibration_fraction": 0.6666666667,
                                    "seed": a.seed, "stratify_by": "major_category"}, config_path="x")
    same = {s.sample_id for s in e2} == {s.sid for s in ev}
    print("evaluation ids identical to authors' split function:", same)
