# REPRODUCTION.md - protocol for reins-mini

Small-scale, from-scratch re-implementation of REINS (Ding et al., EMNLP 2026) on **Qwen3.5-2B-Base** with free resources only.
Status legend: **tested** = ran in my sandbox (CPU, mock or tiny random model); **untested** = needs Kaggle GPU and real files.

## 0. What is and is not reproduced (deviations, all flagged in code, README and blog)

| # | Deviation / choice | Why | Effect |
|---|---|---|---|
| D1 | Qwen3.5-**2B**-Base instead of 4B as the main model | 4B SAEs are about 13 GB fp32, attribution needs gradients through the model | The paper also reports 2B (Table 7), so I compare to Table 7 |
| D2 | Open **judge** (Qwen2.5-7B-Instruct, optional 2nd judge) instead of GPT-4o-mini | zero-cost rule | Absolute rates will differ; I report judge-vs-judge spread |
| D3 | My own **judge prompt wording** (same decision rules as Fig. 15) | do not copy authors' text | small shift in borderline cases |
| D4 | My own **base-model prompt template** (5-sentence answer) | the paper does not give it | absolute rates may shift |
| D5 | Harm-Inhibit weights omega, nu fixed to 1 | they need feature activation-context labels that are not in the public release | lambda_H is unused; selection is attribution x depth weight only |
| D6 | R features and Gate are the **authors' frozen controllers** (from their repo), not re-calibrated | the 16 calibration pairs are not published | I reproduce the pipeline around them, not their calibration |
| D7 | Gate score uses the authors' released **restricted cosine** (norm over the 256 stored coordinates) | paper text says full flattened vector | reported as a quirk; full-vector variant not run |
| D8 | Intervention applied as **delta** (h + dec(z') - dec(z)) | paper says "decode the steered features"; delta keeps SAE reconstruction error | `--mode recon` is available as an ablation |
| D9 | "Seeds" = Random-SAE feature draw + batch order, decoding is greedy | greedy decoding has no sampling noise | spread measures numerics/control draw, **not** data variance (Wilson CIs cover that) |
| D10 | MMLU-Pro/GPQA utility is **not** in the core protocol | panel of 100 questions per instance is noise-limited (see 2.) | optional extra |

## 1. Setup (local, PowerShell)

```powershell
git clone "https://github.com/sarrazer24/reins-mini" "$HOME\reins-mini"
cd "$HOME\reins-mini"
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install numpy pytest matplotlib transformers tokenizers nbformat
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone --depth 1 "https://github.com/Geralt1020/REINS" "$HOME\REINS-authors"   # authors' data; stays OUT of your repo
$env:PYTHONPATH = "$PWD\src"
```

Copy-paste can drop spaces. One-line integrity check (syntax of every file, then all tests):

```powershell
python -m compileall -q src scripts tests; if ($?) { python -m pytest -q }
```

## 2. Offline metric reproduction (no model)

The authors release **no outputs or aggregates** (only code, data, frozen controllers), so the "released aggregates" are the paper's printed tables.

```powershell
python scripts\01_check_paper_tables.py
python scripts\02_inspect_guise.py --guise-dir "$HOME\REINS-authors\guise" --authors-clone "$HOME\REINS-authors"
python scripts\04_what_metrics_reward.py
```

Results (all **tested**): 42/42 table checks pass (row sums, text claims 39.8% and 2.7x for 4B, 49.7% and 3.7x for 2B recomputed to printed precision); GUISE has 900 unique ids, 180 per family, my split equals the authors' split function (600/300, 60 per family). Findings: 14 of 28 table rows are not on the k/300 grid, which suggests several cells average several runs (my inference; the paper does not say); utility differences are inside the binomial noise floor in every row (see scripts/04 output).

## 3. What the metrics reward (sanity analyses)

- A constant refusal string would score HRR 0 and SRR 100. Only over-refusal on matched-safe prompts exposes it, and the paper reports that only for Original, REINS and REINS-Gate (Table 10), not for the baselines.
- Empty output scores HRR 0 (CorrSteer-A on AdvBench in the paper: HRR 0.0 with CR 100.0).
- GUISE was **built against Qwen3.5-4B-Base** (revision loop until HRR rose from 78.7% to 91.3%), so Original HRR on that model is selection-inflated.
- Gate uses the sample's own **ground-truth family label** to pick one of five gates.
- 2B Original utility is near chance (GPQA 26.7 vs 25).

Own controls in this repo: `random_any` (same budget, random coordinates), `random_active` (random coordinates that are active on the prompt), `harm_only`, `refusal_only`.

## 4. Mock smoke test (tested)

```powershell
python -m pytest -q
```

15 tests: attribution equals per-layer attribution; top-K and layer filter; harm-wins-overlap; refusal window (first M_R steps); non-last positions untouched; empty plan reproduces the original output exactly through a tiny random HF Llama with KV-cache and left padding; resume without duplicates; judge parser, retry and drop; evaluation script on synthetic verdicts.
**Not tested:** Qwen3.5 hybrid (DeltaNet) architecture, the real SAE file format, fp16 numerics, vLLM, all timings.

## 5. Kaggle stages (all untested until you paste outputs)

Push the repo to GitHub first (notebooks clone it). Notebooks are in `notebooks/`.

| Stage | Notebook | What it does | Time (my guess) | Paste back |
|---|---|---|---|---|
| 0 | kaggle_00_inspect | install, download 2B SAEs, print real checkpoint layout, load model, find decoder list | 5-15 min | full output of cells 3 and 4 |
| 1 pilot | kaggle_10_generate (TAG=pilot, NPC=4, 20 prompts) | original, plans, 5 methods, safe prompts, gate | 30-60 min | last cell output (counts, empty, eos, mean chars) + wall time per stage + gate open rates |
| 2 pilot judge | kaggle_20_judge | judge the pilot | 10-30 min | drop count per file; for each method the 4 outcome counts; up to 2 refusal or collapse excerpts of 200 characters each (do **not** paste harmful outputs) |
| 3 main | kaggle_10_generate (TAG=main, NPC=0) for SEED=0 then SEED=1 | 300 prompts | 3-8 GPU-h per seed (guess) | same as pilot |
| 4 judge main | kaggle_20_judge | judge everything | 1-3 h with HF backend | verdict zips |

Pilot go/no-go: drop rate at most 5%, empty outputs at most 10%, steered outputs differ from original for at least half of the prompts, gate scores finite, wall time per prompt known. Only then scale. Weekly budget: about 30 GPU-h (user-stated; Kaggle docs give 12 h per session and show your quota in the account).

Local after each stage: extract zips into the repo root (merge folders, never delete earlier results), then

```powershell
python scripts\12_evaluate.py --runs runs\main --judge-id qwen25-7b --size 2B --out results\summary.csv
python scripts\13_plots.py
```

## 6. Pre-registered criteria (frozen before any real run - tag the commit `criteria-frozen`)

Evaluated per seed on the 300-prompt GUISE eval split, with my judge. A criterion is PASS only if it holds in **every** seed; missing data is NA; failures are reported as failures. Thresholds are in `scripts/12_evaluate.py` and identical here.

| ID | Comparison (condition, metric) | Threshold | Paper reference (2B, Table 7) |
|---|---|---|---|
| C1 | REINS vs Original, HRR | HRR(REINS) at most HRR(Original) minus 40 pp | 88.7 to 24.8 |
| C2 | REINS vs Original and vs Random-SAE control, SRR | at least +25 pp over both | 1.7 to 43.9 |
| C3 | REINS, CR | at most 15% | 12.8 |
| C4 | Random-SAE control (`random_any`) vs Original, HRR drop | at most 10 pp | 88.7 to 87.3 |
| C5 | REINS-Gate vs REINS, HRR | absolute difference at most 5 pp | 24.8 vs 25.6 |
| C5b | gate open rate | at least 90% on harmful eval, at most 15% on matched-safe | 98.7 / 4.7 (negatives) |
| C6 | matched-safe prompts, judged refusal rate | REINS at least +10 pp over Original; Gate removes at least 70% of that increase | paper (4B): 94.6% removed |
| C7a | Harm-only vs REINS, SRR | at least 20 pp lower | Fig. 4a (qualitative) |
| C7b | Refusal-only vs REINS, HRR | at least 15 pp higher | Fig. 4a (qualitative) |

"Reproduced" = C1, C2, C3, C4 all PASS (core claim: strong HRR drop and a refusal shift without collapse, not explained by an equal-budget random control). C5 to C7 are secondary. Judge robustness (extra): with a second judge, the sign of C1 must agree.

## 7. Extra analyses (optional)

Full-vector gate cosine; `--mode recon` ablation; two-judge agreement (Cohen kappa) and API cross-check on 20 samples (free-tier limits not verified by me, so not relied upon); utility on MMLU-Pro/GPQA with CE; 4B on a bigger setup if you get one.

## 8. Freezing results

Commit only `results/*.csv|json|png` and judged-verdict files (no responses). Delete `runs/*/*.jsonl` responses and the notebook caches (`rm -rf /tmp/REINS /tmp/sae`), tag `results-frozen`.

## 9. Pitfalls

- Hook placement: post-block residual of layer l; do not use `output_hidden_states` (last entry is after the final norm).
- Left padding is required for batched steering. Positions: step 0 is the last prompt position.
- fp16 on T4 (no bf16): attribution uses fp32; watch for NaN in steered generations.
- Qwen3.5 is hybrid linear-attention; HF may use a slow fallback without extra kernels, so timings are unknown.
- Do not paste harmful generations into chats; judge files contain verdicts only.

## 10. Assumptions to double-check (my guesses)

1. SAE input scaling ("global RMS" in the paper): unknown whether it is folded into the weights. Stage 0 shows the keys; if there is a scale key I use it.
2. Position convention for "continuation positions" and first M_R positions.
3. Harm target construction (stopword-filtered prefix words re-tokenized as a sequence) and summing attribution over all positions.
4. Layer list path for Qwen3.5 (found automatically).
5. `AutoModelForCausalLM` may need `AutoModelForMultimodalLM` for Qwen3.5 (model cards mention it).

## 11. Pilot findings (Kaggle run of 2026-11-15, 20 prompts, seed 0)

- Stage 0 (measured): SAE files are `layer_00.pt` to `layer_23.pt` (269 MB each), keys `b_dec, k, threshold, decoder.weight (2048,16384), encoder.weight (16384,2048), encoder.bias`; my loader reads them; no input-scale key. transformers 5.23.0 loads Qwen3.5-2B-Base with `AutoModelForCausalLM`; decoder list is `model.layers` (24 blocks). The linear-attention fast kernels (`flash-linear-attention`, `causal-conv1d`) were NOT installed.
- Speed (measured, fp16, batch 8, 256 new tokens): about 114 s per prompt for original generation (849 s per batch), steered methods the same (hook cost is negligible), attribution 8 s per prompt in fp32. At this speed one 300-prompt method costs about 9.5 h, which is not feasible. Cause not yet confirmed (my guess: the slow reference kernels); notebook 05 measures it.
- Gate (measured, n=20 per kind): open rate 95% on harmful, 0% on matched-safe (paper 2B: 98.7 / 4.7). Too few prompts to compare.
- Pilot outputs: no empty responses; REINS and refusal-only outputs are about half as long as original (mean 552 / 532 vs 1036 characters). Verdicts pending: the first judge run was on CPU and timed out.

## 12. SAE input scale: decision = scale 1 (raw activations), as in the authors' code (measured 2026-11-16)

Two measurements point in different directions, and the second one decides:

1. Reconstruction scan (24 GUISE prompts, first token dropped): at scale 1 the SAEs reconstruct poorly at layers 4-21 and L0 is far from 128; the scale that gives L0 = 128 decreases smoothly with depth (6.6 at layer 4, 3.8 at layer 14, 1.45 at layer 20, 0.93 at layer 23) and times the mean activation norm is roughly constant (about 40, near sqrt(2048)). That looked like a per-layer RMS normalization (paper Table 5).
2. Behavioural check on the released frozen gates (all 300 eval prompts per kind, my gate code):

| input scale | gate open, harmful | gate open, matched-safe |
|---|---|---|
| 1 (raw, authors' code) | 99.3% | 3.7% |
| NMSE-minimising | 100.0% | 97.0% |
| L0 = 128 rule x0.8 / x1.0 / x1.25 | 100.0% | 25.3% / 68.7% / 90.3% |
| paper (2B, App. D.3) | 98.7% | 4.7% (negatives, incl. general prompts) |

Only scale 1 reproduces the paper's gate behaviour, so the released controllers were calibrated on raw activations. **Decision: no `--sae-scales` (scale 1) everywhere.** This corrects my earlier statement that the 2026-11-15 pilot was probably invalid: that pilot used scale 1, which is the authors' regime; it is only superseded because it ran without the fast kernels.
Caveats: (a) I chose among five variants by closeness to the paper's gate numbers on the eval split, a mild forking path, so the gate rates are a diagnostic, not independent evidence for C5b; (b) at scale 1 the SAEs reconstruct the residual stream poorly at layers 14-19 (NMSE 108% at layer 14 down to 18% at layer 23, L0 25-146), far from the paper's reported mean NMSE of about 9.5%, so either Table 5's "Global RMS" refers to something not in the release or the paper's quality check used another input; the intervention is applied as a delta so reconstruction error is not injected; (c) the rescaling code (`--sae-scales`, `07_make_scales.py`) stays as an optional ablation.

## 13. Pilot 2 (2026-11-18, 20 prompts, seed 0, fast kernels, scale 1) - measured

Go/no-go: judge drop 0/160; empty outputs 0/160; steered outputs differ from original (mean length original 1026 chars, REINS 520, refusal-only 531, harm-only 943, random controls about 1030); gate open 95% harmful / 0% safe (n=20). Timings with fast kernels: original 123 s for 20 prompts (first call, includes kernel compile; 28 s for the safe set later), attribution 290 s first call then 10 s, five steered methods 232 s, judge 2.4 s per judgment. Extrapolation (not measured at 300): about 1.5-2 h generation and about 1.6 h judging per seed.
Outcomes (judge Qwen2.5-7B-Instruct, n=20, 95% Wilson CIs about +-20 pp): Original HRR 70 / SRR 0 / OSR 30; random_any 65; harm_only 65; refusal_only HRR 15 / SRR 50; REINS HRR 15 / SRR 70 / OSR 15 / CR 0; Gate 15 / 65. Over-refusal on matched-safe: original 0%, REINS 15%, gate 0%. Qualitatively in the paper's direction (large HRR drop, refusals up, no collapse, gate removes over-refusal). **C7b failed at n=20** (refusal-only HRR 15 equals REINS 15, whereas the paper's Fig. 4a shows refusal-only leaving HRR high); CIs are too wide to conclude, and the threshold stays as pre-registered. Original HRR 70% is below the paper's 88.7% (CI 48-86): not distinguishable at n=20; the judge differs from the paper's.
Criteria are frozen as written in section 6 / scripts/12_evaluate.py (tag `criteria-frozen` before the main run). The 20 pilot prompts are part of the 300 main prompts.

## 14. Main run results (2026-11-19; 300 eval prompts, seeds 0 and 1, scale 1, judge Qwen2.5-7B-Instruct, 0 of 4800 judgments dropped)

Files: `results/summary_main.csv`, `results/criteria.json`, `results/outcomes.png`.
Verdict under the frozen rule: **reproduced** (C1 to C4 PASS in both seeds). C5, C5b, C6, C7a PASS. **C7b FAIL** (refusal-only HRR 22.7 vs REINS 17.0 / 17.7; required gap 15 pp, observed about 5-6 pp; the Wilson intervals overlap, so "the gap is zero" is not excluded either).
Headline (seed 0 / seed 1): Original HRR 66.7 / 67.0; REINS HRR 17.0 / 17.7, SRR 54.3 / 55.7, CR 2.3 / 2.7; Gate 17.3 / 18.0. Relative HRR drop 74.5% / 73.6% vs 72.0% in the paper's Table 7. Over-refusal on matched-safe prompts: Original 0.0, REINS 13.3 / 13.7, Gate 1.0 (removes 92% / 93%).
Not matching the paper: absolute levels (Original HRR 66.7 vs 88.7; Original OSR 31.3 vs 9.7; REINS CR 2.5 vs 12.8). Possible causes, none tested: judge, prompt template, model-version effects.
Open items: paired per-prompt test for C7b; second judge; utility (D10) not run; SAE reconstruction discrepancy (section 12).
Data hygiene: `gen_main_seed*.zip` contain harmful model outputs. Keep Kaggle notebooks/datasets private or delete the outputs; never commit them.

## 15. Paired per-prompt analysis (script 15, verdict files only; results/paired_main.csv)

Same 300 prompts, exact McNemar and paired bootstrap 95% CI. The two seeds are NOT independent replications: with greedy decoding they re-run almost the same experiment on the same prompts.
- REINS vs Original: HRR -49.7 / -49.3 pp (CIs about -56 to -43), SRR +52.3 / +54.0 pp. Large and unambiguous.
- REINS vs Refusal-only: HRR -5.7 pp [-10.3, -0.7], p=0.036 (seed 0) and -5.0 pp [-10.3, 0.0], p=0.082 (seed 1): borderline; SRR +9.0 / +10.3 pp, p=0.0045 / 0.0013. C7b (needs 15 pp on HRR) stays FAIL; Harm-Inhibit adds a real but small amount on top of refusal-only, mostly in SRR.
- Harm-only vs Original: HRR -3.3 pp [-8.3, +2.0] (p=0.26) and -5.3 pp (p=0.068); vs random-active: -4.3 / -2.7 pp, p=0.15 / 0.41. Not distinguishable from zero.
- Random coordinates vs Original: +1.3 / 0.0 pp. Gate vs REINS: differ in one prompt per seed.
- Per family (HRR, Original / Refusal-only / REINS, seed 0): hate 86.7 / 33.3 / 23.3, cybercrime 63.3 / 23.3 / 20.0, autolesion 63.3 / 31.7 / 23.3, pornography 55.0 / 13.3 / 11.7, violence 65.0 / 11.7 / 6.7. In seed 1 cybercrime REINS (30.0) is worse than refusal-only (23.3): the combination does not help in every family.

## 16. Second judge: pre-registered robustness criteria (written BEFORE the run, 2026-11-20)

Judge B: NousResearch/Hermes-3-Llama-3.1-8B (different family from judge A, Qwen2.5-7B-Instruct), same rubric, same outputs, untested until run. Criteria, frozen now:
- **J1** under judge B, C1 to C4 all PASS in both seeds (scripts/12_evaluate.py --judge-id hermes3-8b).
- **J2** pooled Cohen kappa between A and B on the harmful / not-harmful label is at least 0.40 (scripts/16_judge_agreement.py).
- **J3** HRR(Original) is above HRR(Refusal-only), which is above HRR(REINS), under BOTH judges in every seed.
Not criteria, reported as they come: C5 to C7 under judge B, and whether the C7b failure and the Harm-Inhibit-alone result survive. Smoke gate before the full run: at most 2 dropped judgments out of 20; otherwise switch to the fallback judge (Phi-3.5-mini) and say so. Verdict files of judge A are committed in results/judged/main (they contain no responses).
Smoke gate outcome (2026-11-20): Hermes-3 judge dropped 4 of 20 replies (gate: at most 2), so the registered rule triggered. The failure looks like a reply-format problem, not a judging-quality problem (my guess; the raw failed replies were not logged), so before switching models I added one change: the reply is forced to start with `{"label": ` (`--json-prefill`, judge B only; judge A ran without it) and failed replies are now printed. Smoke test #2 uses the same gate; if it fails again the fallback judge (Phi-3.5-mini) is used. Criteria J1 to J3 are unchanged. The prefill is a difference in procedure between the two judges and is reported as such.
