"""GUISE loading and the documented 2:1 stratified split (own implementation).

The GUISE json files are NOT committed here (authors' data, CC BY 4.0): clone
https://github.com/Geralt1020/REINS and point --guise-dir at its `guise/` folder.

Schema quirk found by inspecting the release: the paper's Table 3 lists the fields
`wrapped_harmful_prompt` / `matched_safe_prompt` / `harmful_category`, but the released files use
`wrapped_malicious_prompt` / `borderline_prompt` / `major_category_name` (and the crime family is
named `cybercrime`). We accept both spellings.
"""
from __future__ import annotations
import json, os, random
from dataclasses import dataclass
from typing import Optional

FILES = ("hate.json", "crime.json", "violence.json", "pornography.json", "autolesion.json")
HARM_KEYS = ("wrapped_malicious_prompt", "wrapped_harmful_prompt")
SAFE_KEYS = ("borderline_prompt", "matched_safe_prompt")
CAT_KEYS = ("major_category_name", "harmful_category")


@dataclass(frozen=True)
class Sample:
    sid: str
    harmful: str
    safe: Optional[str]
    category: str


def _first(row, keys):
    for k in keys:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k]
    return None


def load_guise(guise_dir: str) -> list[Sample]:
    out, seen = [], set()
    for fn in FILES:
        path = os.path.join(guise_dir, fn)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {path}")
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            sid, h, s, c = r.get("id"), _first(r, HARM_KEYS), _first(r, SAFE_KEYS), _first(r, CAT_KEYS)
            if not (isinstance(sid, str) and h and c):
                raise ValueError(f"bad row in {fn}: id={sid!r}")
            if sid in seen:
                raise ValueError(f"duplicate id {sid}")
            seen.add(sid)
            out.append(Sample(sid, h, s, c))
    return out


def split_2to1(samples, seed: int = 12, calib_fraction: float = 2 / 3):
    """Per-category: sort by id, shuffle with random.Random(seed), first round(n*(1-f)) -> eval.

    This re-implements the split protocol that the authors' README describes ("fixed 2:1
    major-category-stratified split", preset seed 12). `scripts/02_inspect_guise.py --cross-check`
    compares it to the authors' own function if their clone is present.
    """
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.category, []).append(s)
    calib, evalu = [], []
    rng = random.Random(seed)
    for cat in sorted(groups):
        ordered = sorted(groups[cat], key=lambda s: s.sid)
        rng.shuffle(ordered)
        n_eval = max(1, min(len(ordered) - 1, round(len(ordered) * (1 - calib_fraction))))
        evalu.extend(ordered[:n_eval])
        calib.extend(ordered[n_eval:])
    return calib, evalu
