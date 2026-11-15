# reins-mini

A small-scale, from-scratch re-implementation of **REINS** (Refusal-Enhanced Inhibitory Steering with SAE features, Ding et al., EMNLP 2026) on Qwen3.5-2B-Base, built to run on free Kaggle GPUs. This is my own implementation, not the authors' code; deviations are listed in `REPRODUCTION.md` section 0.

**Status: step 1 done (offline checks + mock tests pass). GPU runs: not yet done - every GPU claim is untested.**

## Layout
```
src/reinsmini/   paper_tables, data, metrics, sae, steer, gate, judge, pipeline
scripts/         01 table checks, 02 inspect GUISE, 03 inspect SAE ckpt, 04 metric sanity,
                 10 GPU stages, 11 judge, 12 evaluate (+ frozen criteria), 13 plots, 14 gate scores
notebooks/       kaggle_00_inspect, kaggle_10_generate, kaggle_20_judge
tests/           15 CPU tests (mock model, tiny HF Llama, judge, evaluation)
results/         small aggregate files only
runs/            raw generations (git-ignored, never commit)
```
## Quick start
See `REPRODUCTION.md`. Not included (license): the authors' code, the GUISE data, SAE checkpoints, model weights. Links: github.com/Geralt1020/REINS, huggingface.co/Carlos4869/REINS-SAE.
GUISE contains harmful prompts for defensive evaluation; do not paste model outputs on harmful prompts into public places.
