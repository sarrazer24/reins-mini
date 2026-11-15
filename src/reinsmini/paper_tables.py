"""Numbers transcribed by hand from the REINS paper (EMNLP 2026 main, Ding et al.).

Source: Table 2 (Qwen3.5-4B-Base), Table 7 (Qwen3.5-2B-Base), Table 6 (judges),
Table 10 (over-refusal) and Appendix D.1/D.3 hyper-parameters. Transcription may contain
typos: `scripts/01_check_paper_tables.py` cross-checks row sums and granularity.
"""
from __future__ import annotations

N_EVAL_GUISE = 300   # paper: 2:1 split of 900 -> 600 calibration / 300 evaluation
N_MMLU_PRO = 70      # utility panel: 70 MMLU-Pro + 30 GPQA questions
N_GPQA = 30

# method -> dict(HRR, SRR, OSR, CR, MMLU, GPQA, CE)
TABLE2_4B = {
    "Original":    dict(HRR=90.3, SRR=3.7,  OSR=5.7,  CR=0.3,  MMLU=50.0, GPQA=40.0, CE=0.0),
    "Random-SAE":  dict(HRR=90.4, SRR=3.3,  OSR=5.3,  CR=1.0,  MMLU=50.0, GPQA=40.0, CE=1.0),
    "Refusal-SAE": dict(HRR=65.0, SRR=2.3,  OSR=13.7, CR=19.0, MMLU=48.0, GPQA=40.0, CE=6.2),
    "SAE-SSV*":    dict(HRR=78.4, SRR=5.0,  OSR=14.3, CR=2.3,  MMLU=48.0, GPQA=38.7, CE=5.6),
    "CorrSteer-A": dict(HRR=43.7, SRR=23.7, OSR=12.3, CR=20.3, MMLU=46.3, GPQA=42.0, CE=16.6),
    "REINS":       dict(HRR=26.3, SRR=63.7, OSR=7.3,  CR=2.7,  MMLU=45.4, GPQA=35.1, CE=26.4),
    "REINS-Gate":  dict(HRR=26.7, SRR=62.7, OSR=8.3,  CR=2.3,  MMLU=49.9, GPQA=40.0, CE=0.1),
}
TABLE7_2B = {
    "Original":    dict(HRR=88.7, SRR=1.7,  OSR=9.7,  CR=0.0,  MMLU=27.1, GPQA=26.7, CE=0.0),
    "Random-SAE":  dict(HRR=87.3, SRR=1.7,  OSR=11.0, CR=0.0,  MMLU=28.6, GPQA=30.0, CE=3.0),
    "Refusal-SAE": dict(HRR=77.7, SRR=0.3,  OSR=11.7, CR=10.3, MMLU=29.1, GPQA=28.7, CE=4.2),
    "SAE-SSV*":    dict(HRR=53.6, SRR=16.7, OSR=28.0, CR=1.7,  MMLU=36.0, GPQA=30.0, CE=33.6),
    "CorrSteer-A": dict(HRR=49.3, SRR=12.0, OSR=11.4, CR=27.3, MMLU=28.0, GPQA=29.3, CE=22.4),
    "REINS":       dict(HRR=24.8, SRR=43.9, OSR=18.6, CR=12.8, MMLU=27.5, GPQA=26.5, CE=41.3),
    "REINS-Gate":  dict(HRR=25.6, SRR=43.2, OSR=18.7, CR=12.6, MMLU=28.0, GPQA=26.7, CE=1.6),
}
# Table 6: REINS safety outcomes on GUISE (Qwen3.5-4B-Base) under three judges
TABLE6_JUDGES = {
    "GPT-4o-mini":       dict(HRR=26.33, SRR=63.67, OSR=7.33,  CR=2.67),
    "Claude Sonnet 4.6": dict(HRR=25.33, SRR=61.00, OSR=10.67, CR=3.00),
    "DeepSeek-V4-Pro":   dict(HRR=20.00, SRR=62.83, OSR=13.67, CR=3.50),
}
# Table 10: matched-safe prompts, Qwen3.5-4B-Base (normal / over-refusal / collapse)
TABLE10_OVERREFUSAL_4B = {
    "Original":   dict(Normal=88.33, OverRefusal=3.67,  Collapse=8.00),
    "REINS":      dict(Normal=66.67, OverRefusal=22.33, Collapse=11.00),
    "REINS-Gate": dict(Normal=88.00, OverRefusal=4.67,  Collapse=7.33),
}
# Hyper-parameters (Appendix D.1 / D.3)
HP = {
    "2B": dict(layers=24, hidden=2048, sae_width=16384, harm_layer_min=14, K_H=8,
               lambda_H=0.35, K_R=8, alpha_R=5.0, M_R=8, gate_topk=256, beta=0.10,
               gate_open_harmful=98.7, gate_open_negative=4.7),
    "4B": dict(layers=32, hidden=2560, sae_width=20480, harm_layer_min=14, K_H=12,
               lambda_H=1.0, K_R=2, alpha_R=5.5, M_R=8, gate_topk=256, beta=0.10,
               gate_open_harmful=99.3, gate_open_negative=3.5),
}
TEXT_CLAIMS = {
    "4B_HRR_reduction_vs_CorrSteerA_pct": 39.8,
    "4B_SRR_ratio_vs_CorrSteerA": 2.7,
    "2B_HRR_reduction_vs_CorrSteerA_pct": 49.7,
    "2B_SRR_ratio_vs_CorrSteerA": 3.7,
}
