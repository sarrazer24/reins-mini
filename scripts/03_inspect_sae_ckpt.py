"""Stage 0 on Kaggle: print the real layout of the SAE checkpoints and test my loader on them.
Paste the full output back to me."""
import argparse, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import sae as SAE

ap = argparse.ArgumentParser()
ap.add_argument("--sae-root", required=True, help="e.g. <snapshot>/qwen3.5-2b-base")
ap.add_argument("--layers", default="0,14,23")
ap.add_argument("--d-model", type=int, default=2048)
ap.add_argument("--width", type=int, default=16384)
a = ap.parse_args()
print("files:", sorted(os.listdir(a.sae_root))[:40])
for l in [int(x) for x in a.layers.split(",")]:
    p = os.path.join(a.sae_root, f"layer_{l:02d}.pt")
    print(f"\n== layer {l}: {p} ({os.path.getsize(p)/1e6:.0f} MB)")
    info = SAE.describe_checkpoint(p)
    print(info["type"]); [print("  ", k, v) for k, v in info["keys"].items()]
    try:
        s = SAE.load_sae(p, a.d_model, a.width)
        x = torch.randn(4, a.d_model) * 3
        z = s.encode(x); r = s.decode(z)
        print("loader OK: n,d =", s.n, s.d, "threshold =", s.threshold, "input_scale =", s.input_scale,
              "| L0 on random input:", float((z > 0).sum(-1).float().mean()))
    except Exception as e:
        print("LOADER FAILED:", repr(e))
