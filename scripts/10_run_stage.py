"""GPU stages (Kaggle). One command per stage; every stage appends to JSONL and is resumable.
  --stage original : unsteered generations (control + source of Harm-Inhibit prefixes)
  --stage plans    : per-prompt Harm-Inhibit attribution (use --dtype float32 for best gradient precision)
  --stage steer    : steered generations for --methods (reins harm_only refusal_only random_any random_active)
"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import data as D, pipeline as P, steer as S, sae as SAE, paper_tables as T

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["original", "plans", "steer"])
ap.add_argument("--size", default="2B", choices=["2B", "4B"])
ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
ap.add_argument("--sae-root", required=True)
ap.add_argument("--controllers", required=True, help="authors' configs/qwen35_2b_guise.controllers.json")
ap.add_argument("--guise-dir", required=True)
ap.add_argument("--out", required=True, help="e.g. runs/pilot/seed0")
ap.add_argument("--field", default="harmful", choices=["harmful", "safe"])
ap.add_argument("--split-seed", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-per-category", type=int, default=0, help="pilot: first N eval prompts per category (0 = all 60)")
ap.add_argument("--methods", nargs="*", default=["reins"])
ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--mode", default="delta", choices=["delta", "recon"])
a = ap.parse_args()

hp = T.HP[a.size]
samples = D.load_guise(a.guise_dir)
_, ev = D.split_2to1(samples, seed=a.split_seed)
if a.n_per_category:
    cnt, keep = {}, []
    for s in sorted(ev, key=lambda s: s.sid):
        if cnt.get(s.category, 0) < a.n_per_category:
            keep.append(s); cnt[s.category] = cnt.get(s.category, 0) + 1
    ev = keep
print(f"eval prompts: {len(ev)}  field={a.field}  stage={a.stage}")
os.makedirs(a.out, exist_ok=True)
suffix = "" if a.field == "harmful" else "_safe"
f_orig, f_plans = f"{a.out}/original{suffix}.jsonl", f"{a.out}/plans{suffix}.jsonl"

from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
dt = dict(float16=torch.float16, float32=torch.float32, bfloat16=torch.bfloat16)[a.dtype]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
lname, layers = S.find_layer_list(model, hp["layers"])
print("decoder layers found at:", lname, "| dtype", a.dtype)

ctrl = json.load(open(a.controllers))["controllers"]
refusal = {(f["layer"], f["feature_idx"]): f["additive_value"] for f in ctrl["r_features"]}

def load_saes(ls):
    out = {}
    for l in sorted(ls):
        out[l] = SAE.load_sae(os.path.join(a.sae_root, f"layer_{l:02d}.pt"), hp["hidden"], hp["sae_width"]).to_(dev, torch.float32)
    return out

t0 = time.time()
if a.stage == "original":
    P.run_original(model, tok, ev, f_orig, field=a.field, bs=a.bs, seed=a.seed, device=dev)
elif a.stage == "plans":
    saes = load_saes(range(hp["harm_layer_min"], hp["layers"]))
    P.run_plans(model, tok, saes, layers, ev, f_orig, f_plans, hp, device=dev, seed=a.seed, field=a.field)
else:
    need = {l for l, _ in refusal} | set(range(hp["harm_layer_min"], hp["layers"]))
    saes = load_saes(need)
    for m in a.methods:
        P.run_steered(model, tok, saes, layers, ev, f_plans, f"{a.out}/{m}{suffix}.jsonl", m, refusal, hp,
                      bs=a.bs, seed=a.seed, device=dev, mode=a.mode, field=a.field)
print(f"stage {a.stage} done in {time.time()-t0:.0f}s")
