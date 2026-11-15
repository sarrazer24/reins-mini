"""REINS-Gate scoring.

Paper Eq. 8: s = cos(v(x), w_gate), v(x) = flattened mean SAE feature vector over prompt tokens.
Quirk found in the authors' release: their scorer computes the input norm ONLY over the gate's 256
stored coordinates (a restricted cosine), and the stored thresholds are mostly negative. We implement
both variants and report both: `restricted` (matches the release) and `full` (matches the paper text;
needs all-layer SAE norms).
"""
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple


def gate_score_restricted(entries: Sequence[Tuple[int, int, float]], means: Dict[int, "object"], norm: float) -> float:
    """entries: (layer, feature, weight). means[l]: (n,) mean SAE features over prompt tokens."""
    dot = sq = 0.0
    for l, j, w in entries:
        v = float(means[l][j]) if l in means else 0.0
        dot += v * w
        sq += v * v
    n = math.sqrt(sq)
    return 0.0 if (n <= 0 or norm <= 0) else dot / (n * norm)


def gate_score_full(entries, means: Dict[int, "object"], norm: float) -> float:
    dot = 0.0
    for l, j, w in entries:
        dot += float(means[l][j]) * w if l in means else 0.0
    sq = sum(float((m.float() ** 2).sum()) for m in means.values())
    n = math.sqrt(sq)
    return 0.0 if (n <= 0 or norm <= 0) else dot / (n * norm)
