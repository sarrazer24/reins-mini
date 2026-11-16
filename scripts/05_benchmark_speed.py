"""Stage 0b (Kaggle GPU): where does the time go, and are the SAEs fed activations in the right scale?
1) decode speed per step at batch 1 and 8 (the pilot measured about 3.3 s per decode step: 849 s for 256 steps at batch 8)
2) are the fast Qwen3.5 kernels (flash-linear-attention, causal-conv1d) installed?
3) real-activation check of the SAEs: mean L0 should be close to 128 (BatchTopK target) and NMSE near 10% if the input scale is right."""
import argparse, os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import data as D, pipeline as P, steer as S, sae as SAE

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base"); ap.add_argument("--sae-root", required=True)
ap.add_argument("--guise-dir", required=True); ap.add_argument("--layers", default="14,20,23")
a = ap.parse_args()
for mod in ("fla", "causal_conv1d"):
    try: __import__(mod); print(f"{mod}: installed")
    except Exception as e: print(f"{mod}: NOT installed ({type(e).__name__})")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("gpus:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
ev = D.split_2to1(D.load_guise(a.guise_dir))[1][:8]
texts = [P.render(s.harmful) for s in ev]

def timed(bs, n):
    enc = tok(texts[:bs], return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(): m.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    torch.cuda.synchronize(); return time.time() - t
timed(1, 4)  # warm-up
for bs in (1, 8):
    t8, t40 = timed(bs, 8), timed(bs, 40)
    print(f"batch {bs}: 8 tokens {t8:.1f}s, 40 tokens {t40:.1f}s -> {(t40-t8)/32:.2f} s per decode step, prefill+overhead about {t8-8*(t40-t8)/32:.1f}s")

_, layers = S.find_layer_list(m, 24)
cap, hs = {}, []
for l in [int(x) for x in a.layers.split(",")]:
    hs.append(layers[l].register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach())))
with torch.no_grad():
    for t in texts[:4]: m(**tok(t, return_tensors="pt").to("cuda"))
    # keep the last prompt only for the SAE check (cap holds the last forward)
for h in hs: h.remove()
for l, x in cap.items():
    s = SAE.load_sae(os.path.join(a.sae_root, f"layer_{l:02d}.pt"), 2048, 16384).to_("cuda", torch.float32)
    x = x[0].float(); z = s.encode(x); rec = s.decode(z)
    nmse = 100 * float(((x - rec) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    print(f"layer {l}: tokens {x.shape[0]}  mean L0 {float((z>0).sum(-1).float().mean()):.1f} (target ~128)  NMSE {nmse:.1f}% (paper avg ~9.5)  mean |x| {float(x.norm(dim=-1).mean()):.1f}")
