"""Generation pipeline: original -> plans (attribution) -> steered generation. Disk-cached, resumable.

All records are JSONL, one line per (method, sid, seed). Raw caches go under runs/ (git-ignored).
"""
from __future__ import annotations
import json, os, random, time
from typing import Dict, List, Optional, Sequence

import torch

from . import steer as S

# GUESS: wording is mine; the paper does not give the base-model prompt format.
TEMPLATE = ("Answer the following question directly, in about five complete sentences, "
            "without restating it.\n\nQuestion: {q}\n\nAnswer:")
GEN = dict(max_new_tokens=256, do_sample=False, repetition_penalty=1.15, no_repeat_ngram_size=4)  # authors' preset values


def render(q: str) -> str:
    return TEMPLATE.format(q=q)


def read_jsonl(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path: str, rows: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def done_ids(path: str, key="sid") -> set:
    return {r[key] for r in read_jsonl(path)}


@torch.no_grad()
def generate_batch(model, tok, questions: Sequence[str], steerer=None, gen: Optional[dict] = None, device="cuda"):
    gen = dict(GEN, **(gen or {}))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok([render(q) for q in questions], return_tensors="pt", padding=True).to(device)
    ctx = steerer if steerer is not None else _Null()
    with ctx:
        out = model.generate(**enc, pad_token_id=tok.pad_token_id, **gen)
    texts, finished = [], []
    for row in out[:, enc["input_ids"].shape[1]:]:
        ids = row.tolist()
        eos = tok.eos_token_id is not None and tok.eos_token_id in ids
        texts.append(tok.decode(ids, skip_special_tokens=True).strip())
        finished.append(bool(eos))
    return texts, finished


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def run_original(model, tok, samples, out_path, field="harmful", bs=8, seed=0, device="cuda", log=print):
    todo = [s for s in samples if s.sid not in done_ids(out_path)]
    rng = random.Random(seed); rng.shuffle(todo)          # seed perturbs batch composition (fp16 numerics)
    t0 = time.time()
    for i in range(0, len(todo), bs):
        b = todo[i:i + bs]
        texts, fin = generate_batch(model, tok, [getattr(s, field) for s in b], None, device=device)
        append_jsonl(out_path, [dict(sid=s.sid, method="original", seed=seed, field=field, response=t, eos=f)
                                for s, t, f in zip(b, texts, fin)])
        log(f"original {i + len(b)}/{len(todo)}  {time.time() - t0:.0f}s")


def run_plans(model, tok, saes, layers, samples, orig_path, out_path, hp, device="cuda", seed=0, log=print, field="harmful"):
    """Attribution per prompt (batch size 1). Needs `saes` for layers >= hp['harm_layer_min']."""
    orig = {r["sid"]: r for r in read_jsonl(orig_path)}
    have = done_ids(out_path)
    lmax = hp["layers"] - 1
    for n, s in enumerate(x for x in samples if x.sid not in have):
        resp = orig[s.sid]["response"]
        tok_ids = tok(resp, add_special_tokens=False)["input_ids"][:32]
        pieces = [tok.decode([t]) for t in tok_ids]
        m = S.prefix_length(pieces, 32)
        prefix = tok.decode(tok_ids[:m])
        q = getattr(s, field)
        target = S.harm_target_text(q, prefix)
        p_ids = tok(render(q), return_tensors="pt")["input_ids"][0].tolist()
        t_ids = tok(" " + target, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + t_ids], device=device)
        A, act = S.attribution_pass(model, layers, saes, ids, len(p_ids), device)
        harm = S.select_harm_features(A, lmax, hp["harm_layer_min"], hp["K_H"])
        rng = random.Random(f"{seed}:{s.sid}")
        pool = [(l, int(j)) for l, a in act.items() for j in a.nonzero().flatten().tolist()]
        rnd_active = rng.sample(pool, min(len(pool), hp["K_H"] + hp["K_R"]))
        rnd_any = [(rng.choice(sorted(saes)), rng.randrange(hp["sae_width"])) for _ in range(hp["K_H"] + hp["K_R"])]
        append_jsonl(out_path, [dict(sid=s.sid, harm=harm, random_active=rnd_active, random_any=rnd_any,
                                     prefix_tokens=m, target=target, n_active=len(pool))])
        log(f"plans {n + 1}  harm={len(harm)}")


def run_steered(model, tok, saes, layers, samples, plans_path, out_path, method, refusal, hp, bs=8, seed=0,
                device="cuda", mode="delta", log=print, field="harmful"):
    """method in {reins, harm_only, refusal_only, random_any, random_active}."""
    plans = {r["sid"]: r for r in read_jsonl(plans_path)}
    todo = [s for s in samples if s.sid not in done_ids(out_path)]
    rng = random.Random(seed); rng.shuffle(todo)
    t0 = time.time()
    for i in range(0, len(todo), bs):
        b = todo[i:i + bs]
        rows = []
        for s in b:
            p = plans[s.sid]
            if method in ("reins", "harm_only"):
                rows.append(S.RowPlan(harm=[tuple(x) for x in p["harm"]]))
            elif method == "refusal_only":
                rows.append(S.RowPlan(harm=[]))
            else:
                rows.append(S.RowPlan(harm=[tuple(x) for x in p[method]]))
        ref = refusal if method in ("reins", "refusal_only") else {}
        layer_set = {l for r in rows for l, _ in r.harm} | {l for l, _ in ref}
        sub = {l: saes[l] for l in layer_set}
        st = S.Steerer(layers, sub, rows, ref, hp["M_R"], hp["sae_width"], mode=mode)
        texts, fin = generate_batch(model, tok, [getattr(s, field) for s in b], st, device=device)
        append_jsonl(out_path, [dict(sid=s.sid, method=method, seed=seed, response=t, eos=f) for s, t, f in zip(b, texts, fin)])
        log(f"{method} {i + len(b)}/{len(todo)}  {time.time() - t0:.0f}s")
