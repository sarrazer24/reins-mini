"""Stage 0c (Kaggle GPU): is the SAE fed activations at the scale it was trained on?
Benchmark 05 found NMSE 94% / L0 25 at layer 14 (paper: NMSE ~9.5%, L0 ~128). Two candidate explanations:
 (1) the first token of each prompt has a huge 'sink' activation that dominates NMSE, or
 (2) the checkpoints expect inputs rescaled by a per-layer constant (paper Table 5: 'Global RMS' normalisation).
This scan drops token 0, scans the input scale c over a grid for every layer, and writes
results/sae_scale_scan.csv and results/sae_scales_candidate.json (c with minimum NMSE per layer; I review it before use)."""
import argparse, csv, json, math, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import data as D, pipeline as P, steer as S, sae as SAE

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base"); ap.add_argument("--sae-root", required=True)
ap.add_argument("--guise-dir", required=True); ap.add_argument("--n-prompts", type=int, default=24)
ap.add_argument("--out", default="results")
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
_, layers = S.find_layer_list(m, 24)
ev = sorted(D.split_2to1(D.load_guise(a.guise_dir))[1], key=lambda s: s.sid)[:: max(1, 300 // a.n_prompts)][: a.n_prompts]
H = {l: [] for l in range(24)}; first = {l: [] for l in range(24)}
hs = [layers[l].register_forward_hook(lambda mod, i, o, l=l: H[l].append((o[0] if isinstance(o, tuple) else o)[0].detach().float().cpu())) for l in range(24)]
with torch.no_grad():
    for s in ev: m(**tok(P.render(s.harmful), return_tensors="pt").to("cuda"))
for h in hs: h.remove()
del m; torch.cuda.empty_cache()
scales = [2 ** (k / 2) for k in range(-8, 9)]   # 0.0625 ... 16
rows, cand = [], {}
print(f"{len(ev)} prompts, tokens/layer (excluding token 0): {sum(h.shape[0]-1 for h in H[0])}")
print("layer | mean|x| tok>0 | |x| tok0 | RMS-hyp c | NMSE@c=1 L0@c=1 | best c  NMSE@best L0@best | NMSE@hyp L0@hyp")
for l in range(24):
    x = torch.cat([h[1:] for h in H[l]]).cuda()
    n0 = float(torch.stack([h[0].norm() for h in H[l]]).mean()); mn = float(x.norm(dim=-1).mean())
    sae = SAE.load_sae(os.path.join(a.sae_root, f"layer_{l:02d}.pt"), 2048, 16384).to_("cuda", torch.float32)
    hyp = math.sqrt(2048) / mn
    res = SAE.scale_scan(x, sae, sorted(scales + [1.0, hyp]))
    for c, nm, l0 in res: rows.append(dict(layer=l, scale=round(c, 4), nmse=round(nm, 2), l0=round(l0, 1)))
    by = {round(c, 4): (nm, l0) for c, nm, l0 in res}
    best = min(res, key=lambda r: r[1]); cand[str(l)] = round(best[0], 4)
    print(f"{l:5d} | {mn:13.1f} | {n0:8.1f} | {hyp:9.2f} | {by[1.0][0]:7.1f}% {by[1.0][1]:6.1f} | {best[0]:6.2f} {best[1]:9.1f}% {best[2]:7.1f} | {by[round(hyp,4)][0]:6.1f}% {by[round(hyp,4)][1]:6.1f}")
    del sae
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "sae_scale_scan.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer", "scale", "nmse", "l0"]); w.writeheader(); w.writerows(rows)
json.dump(cand, open(os.path.join(a.out, "sae_scales_candidate.json"), "w"), indent=1)
print("wrote results/sae_scale_scan.csv and results/sae_scales_candidate.json")
