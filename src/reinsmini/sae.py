"""Minimal BatchTopK-SAE inference module + tolerant checkpoint loader (UNTESTED on the real files).

I could not download the authors' SAE bundle in my sandbox (huggingface.co is not reachable), so the
loader detects the layout from key names and shapes, and `scripts/03_inspect_sae_ckpt.py` prints what
is really inside a file. If the real keys differ, that script's output tells us what to change.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SparseAE(nn.Module):
    """x -> relu((x - b_dec) W_enc^T + b_enc) [-> threshold gate] -> z ; z -> z W_dec^T + b_dec."""

    def __init__(self, enc_w, enc_b, dec_w, b_dec, threshold=None, input_scale=None):
        super().__init__()
        self.register_buffer("enc_w", enc_w)        # (n, d)
        self.register_buffer("enc_b", enc_b)        # (n,)
        self.register_buffer("dec_w", dec_w)        # (d, n)
        self.register_buffer("b_dec", b_dec)        # (d,)
        self.threshold = None if threshold is None else float(threshold)
        self.input_scale = None if input_scale is None else float(input_scale)
        self.n, self.d = enc_w.shape

    @torch.no_grad()
    def to_(self, device=None, dtype=None):
        for k in ("enc_w", "enc_b", "dec_w", "b_dec"):
            setattr(self, k, getattr(self, k).to(device=device, dtype=dtype))
        return self

    def encode(self, x):
        if self.input_scale is not None:
            x = x * self.input_scale
        f = torch.relu((x - self.b_dec) @ self.enc_w.T + self.enc_b)
        if self.threshold is not None and self.threshold > 0:
            f = f * (f > self.threshold)
        return f

    def decode(self, z):
        out = z @ self.dec_w.T + self.b_dec
        return out / self.input_scale if self.input_scale is not None else out


_ENC_W = ("encoder.weight", "W_enc", "enc.weight", "encoder.W")
_ENC_B = ("encoder.bias", "b_enc", "enc.bias")
_DEC_W = ("decoder.weight", "W_dec", "dec.weight", "decoder.W")
_B_DEC = ("b_dec", "decoder.bias", "dec.bias", "bias_dec")


def _find(sd, names):
    for k in names:
        if k in sd:
            return sd[k]
    return None


def describe_checkpoint(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    info = {"type": type(obj).__name__, "keys": {}}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                info["keys"][k] = f"tensor{tuple(v.shape)} {v.dtype}"
            elif isinstance(v, dict):
                info["keys"][k] = "dict[" + ", ".join(list(v.keys())[:8]) + "]"
            else:
                info["keys"][k] = f"{type(v).__name__}: {str(v)[:60]}"
    return info


def load_sae(path: str, d_model: int | None = None, n_features: int | None = None) -> SparseAE:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj
    for wrap in ("state_dict", "model", "ae", "sae"):
        if isinstance(sd, dict) and wrap in sd and isinstance(sd[wrap], dict):
            sd = sd[wrap]
    if not isinstance(sd, dict):
        raise ValueError(f"unsupported checkpoint object {type(obj).__name__} (run 03_inspect_sae_ckpt.py)")
    ew, eb, dw, bd = (_find(sd, k) for k in (_ENC_W, _ENC_B, _DEC_W, _B_DEC))
    if ew is None or dw is None:
        raise KeyError(f"could not find encoder/decoder weights; keys={list(sd)[:20]}")
    if eb is None:
        eb = torch.zeros(ew.shape[0] if ew.shape[0] != d_model else ew.shape[1])
    # orient: enc_w -> (n, d); dec_w -> (d, n)
    n = n_features or max(ew.shape)
    d = d_model or min(ew.shape)
    if ew.shape == (d, n):
        ew = ew.T
    if dw.shape == (n, d):
        dw = dw.T
    assert ew.shape == (n, d) and dw.shape == (d, n), (ew.shape, dw.shape, d, n)
    if bd is None:
        bd = torch.zeros(d)
    thr = sd.get("threshold", None)
    if torch.is_tensor(thr):
        thr = float(thr.reshape(-1)[0])
    scale = sd.get("input_scale", sd.get("activation_scale", None))
    if torch.is_tensor(scale):
        scale = float(scale.reshape(-1)[0])
    return SparseAE(ew.float().contiguous(), eb.float(), dw.float().contiguous(), bd.float(), thr, scale)
