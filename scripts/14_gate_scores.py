"""Gate scores (restricted cosine, as in the authors' release) for harmful and matched-safe eval prompts.
Uses the sample's own major category to pick the gate (the release does this too - an oracle label)."""
import argparse, csv, json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import data as D, pipeline as P, steer as S, sae as SAE, gate as G, paper_tables as T

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="2B"); ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
ap.add_argument("--sae-root", required=True); ap.add_argument("--controllers", required=True)
ap.add_argument("--guise-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sae-scales", default=None)
ap.add_argument("--split-seed", type=int, default=12); ap.add_argument("--n-per-category", type=int, default=0)
a = ap.parse_args()
hp = T.HP[a.size]
ev = D.split_2to1(D.load_guise(a.guise_dir), seed=a.split_seed)[1]
if a.n_per_category:
    c, k = {}, []
    for s in sorted(ev, key=lambda s: s.sid):
        if c.get(s.category, 0) < a.n_per_category: k.append(s); c[s.category] = c.get(s.category, 0) + 1
    ev = k
gates = json.load(open(a.controllers))["controllers"]["gates_by_major_category"]
layers_needed = sorted({e["layer"] for g in gates.values() for e in g["entries"]})
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()
_, layers = S.find_layer_list(model, hp["layers"])
saes = {l: SAE.load_sae(os.path.join(a.sae_root, f"layer_{l:02d}.pt"), hp["hidden"], hp["sae_width"]).to_(dev, torch.float32) for l in layers_needed}
SAE.apply_scales(saes, a.sae_scales)
rows = []
with torch.no_grad():
    for s in ev:
        for kind, text in (("harmful", s.harmful), ("safe", s.safe)):
            cap = {}
            hs = [layers[l].register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o))) for l in layers_needed]
            model(**tok(P.render(text), return_tensors="pt").to(dev)); [h.remove() for h in hs]
            means = {l: saes[l].encode(cap[l][0].float()).mean(0).cpu() for l in layers_needed}
            g = gates[s.category]
            ent = [(e["layer"], e["feature_idx"], e["weight"]) for e in g["entries"]]
            sc = G.gate_score_restricted(ent, means, g["norm"])
            rows.append(dict(sid=s.sid, kind=kind, category=s.category, score=sc, threshold=g["threshold"], open=int(sc >= g["threshold"])))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for kind in ("harmful", "safe"):
    r = [x["open"] for x in rows if x["kind"] == kind]
    sc_ = [x["score"] for x in rows if x["kind"] == kind]
    print(f"gate score mean {sum(sc_)/len(sc_):.3f} |", end=" ")
    print(f"gate open rate on {kind}: {100*sum(r)/len(r):.1f}% (n={len(r)}); paper (2B): harmful 98.7 / negatives 4.7")
