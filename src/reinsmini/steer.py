"""REINS mechanics, my own implementation from the paper text (Sec. 5, App. D.1).

Pieces: (1) layer weights and Harm-Inhibit selection, (2) gradient x activation attribution with a
straight-through SAE patch, (3) the joint intervention hook (zero harmful features at every continuation
position; add refusal values on the first M_R positions; harm wins overlaps).

Deviations / guesses are marked `GUESS:` and listed in REPRODUCTION.md.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import torch

STOP = frozenset("""a an the and or but if then of to in on at by for with from as is are was were be been being it its this that
these those i you he she we they me my your our their him her them do does did not no so than too very can could should would will
may might must have has had what which who whom how when where why about into over under again further once here there all any both
each few more most other some such only own same just also""".split())


def layer_depth(l: int, lmax: int) -> float:
    return min(max(l / lmax, 0.0), 1.0)


def harm_eta(l: int, lmax: int, omega: float = 1.0, nu: float = 1.0) -> float:
    """eta^H = (0.25 + 1.10 rho_l) * min(omega, nu). GUESS: omega = nu = 1 because the semantic overlay
    (feature activation-context labels) that defines them is not in the public release."""
    return (0.25 + 1.10 * layer_depth(l, lmax)) * min(omega, nu)


def select_harm_features(attr: Dict[int, torch.Tensor], lmax: int, layer_min: int, k: int) -> List[Tuple[int, int]]:
    """attr[l] is a (n,) tensor of summed grad x activation. q = relu(A) * eta; top-k over layers >= layer_min."""
    cands = []
    for l, a in attr.items():
        if l < layer_min:
            continue
        q = torch.relu(a.float()) * harm_eta(l, lmax)
        kk = min(k, q.numel())
        v, i = torch.topk(q, kk)
        cands.extend((float(x), l, int(j)) for x, j in zip(v, i) if float(x) > 0)
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(l, j) for _, l, j in cands[:k]]


def prefix_length(token_texts: Sequence[str], max_len: int = 32) -> int:
    """Earliest m <= max_len whose decoded prefix ends at a sentence or line boundary; else min(max_len, len)."""
    acc = ""
    for m, t in enumerate(token_texts[:max_len], start=1):
        acc += t
        if acc.rstrip(" ").endswith((".", "!", "?", "\n")) or acc.endswith("\n"):
            return m
    return min(max_len, len(token_texts))


def harm_target_text(prompt: str, prefix: str) -> str:
    """GUESS (paper D.1): keep prefix words that also occur in the prompt after stopword removal;
    if empty, stopword-filter the prefix alone; if still empty, fall back to the raw prefix."""
    words = lambda s: re.findall(r"[A-Za-z0-9']+", s)
    pw = {w.lower() for w in words(prompt) if w.lower() not in STOP}
    kept = [w for w in words(prefix) if w.lower() in pw]
    if not kept:
        kept = [w for w in words(prefix) if w.lower() not in STOP]
    return " ".join(kept) if kept else prefix


def _primary(output):
    if isinstance(output, tuple):
        return output[0], (lambda h, o=output: (h,) + tuple(o[1:]))
    return output, (lambda h: h)


def find_layer_list(model, n_layers: int):
    """Find the decoder ModuleList (Qwen3.5 may nest it under model.language_model)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers:
            best = best or (name, mod)
            if name.endswith("layers"):
                return name, mod
    if best is None:
        raise RuntimeError(f"no ModuleList with {n_layers} blocks found")
    return best


def attribution_pass(model, layers, saes: Dict[int, "object"], input_ids, target_start: int,
                     device) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Straight-through patch at every layer in `saes`: forward value == original residual, backward
    flows through SAE features only. Returns (A[l] (n,), active[l] (n,) bool).
    A[l,j] = sum_t z[l,t,j] * dJ/dz[l,t,j], J = sum of target-token log-probs (teacher forced).
    z is computed from the DETACHED residual so no gradient flows through the encoder."""
    zs, handles = {}, []

    def mk(l):
        def hook(_m, _i, out):
            h, rebuild = _primary(out)
            sae = saes[l]
            z = sae.encode(h.detach().to(sae.enc_w.dtype)).detach().requires_grad_(True)
            rec = sae.decode(z).to(h.dtype)
            zs[l] = z
            return rebuild(h + (rec - rec.detach()))
        return hook

    for l in saes:
        handles.append(layers[l].register_forward_hook(mk(l)))
    try:
        with torch.enable_grad():
            logits = model(input_ids=input_ids, return_dict=True).logits
            lp = torch.log_softmax(logits.float(), dim=-1)
            T = input_ids.shape[1]
            pos = torch.arange(target_start, T, device=input_ids.device)
            J = lp[0, pos - 1, :].gather(-1, input_ids[0, pos].unsqueeze(-1)).sum()
            grads = torch.autograd.grad(J, [zs[l] for l in saes], allow_unused=True)
    finally:
        for h in handles:
            h.remove()
    A, act = {}, {}
    for (l, z), g in zip(zs.items(), grads):
        g = torch.zeros_like(z) if g is None else g
        A[l] = (z.detach() * g).sum(dim=(0, 1)).float().cpu()
        act[l] = (z.detach() > 0).any(dim=(0, 1)).cpu()
    return A, act


@dataclass
class RowPlan:
    harm: List[Tuple[int, int]] = field(default_factory=list)     # (layer, feature) zeroed at all positions


class Steerer:
    """Context manager that installs the joint intervention on a decoder layer list.

    Position convention (GUESS, consistent with an 'event' counter): the forward call that processes the
    prompt counts as continuation step 0 and modifies only the LAST prompt position (it produces the first
    generated token); each later decode call modifies its single new position. Refusal values are added
    while step < M_R. Use batch generation with LEFT padding.
    """

    def __init__(self, layers, saes: Dict[int, object], plans: Sequence[RowPlan],
                 refusal: Dict[Tuple[int, int], float], m_r: int, n_features: int, mode: str = "delta"):
        self.layers, self.saes, self.m_r, self.mode = layers, saes, m_r, mode
        self.B = len(plans)
        self.n = n_features
        dev = next(iter(saes.values())).enc_w.device if saes else "cpu"
        self.zero_mask, self.add_mat, self.handles, self.calls = {}, {}, [], {}
        lset = {l for p in plans for l, _ in p.harm} | {l for l, _ in refusal}
        for l in sorted(lset):
            zm = torch.zeros(self.B, self.n, dtype=torch.bool, device=dev)
            for b, p in enumerate(plans):
                for (ll, j) in p.harm:
                    if ll == l:
                        zm[b, j] = True
            add = torch.zeros(self.B, self.n, device=dev)
            for (ll, j), v in refusal.items():
                if ll == l:
                    add[:, j] = v
            add = add.masked_fill(zm, 0.0)            # harm wins on overlaps
            self.zero_mask[l], self.add_mat[l] = zm, add
            self.calls[l] = 0
        self.trace = {"events": 0, "rel_norm_sum": 0.0}

    def reset(self):
        for l in self.calls:
            self.calls[l] = 0

    def _hook(self, l):
        def hook(_m, _i, out):
            h, rebuild = _primary(out)
            step = self.calls[l]
            self.calls[l] += 1
            sae = self.saes[l]
            x = h[:, -1, :]
            xf = x.to(sae.enc_w.dtype)
            z = sae.encode(xf)
            z2 = z.masked_fill(self.zero_mask[l][: x.shape[0]], 0.0)
            if step < self.m_r:
                z2 = z2 + self.add_mat[l][: x.shape[0]]
            if self.mode == "delta":
                new = xf + (sae.decode(z2) - sae.decode(z))        # preserves SAE reconstruction error
            else:
                new = sae.decode(z2)                                # literal "decode the steered features"
            h2 = h.clone()
            h2[:, -1, :] = new.to(h.dtype)
            self.trace["events"] += 1
            self.trace["rel_norm_sum"] += float(((new - xf).detach().norm(dim=-1) / (xf.norm(dim=-1) + 1e-6)).mean())
            return rebuild(h2)
        return hook

    def __enter__(self):
        self.reset()
        for l in self.zero_mask:
            self.handles.append(self.layers[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
