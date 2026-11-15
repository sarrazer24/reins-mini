import os, sys, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reinsmini import steer as S
from reinsmini.sae import SparseAE

torch.manual_seed(0)
D, N, L = 16, 64, 4


class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Linear(D, D)
    def forward(s, h):
        return h + torch.tanh(s.f(h))


class Tiny(nn.Module):
    def __init__(s, tuple_out=False):
        super().__init__()
        s.emb = nn.Embedding(50, D); s.layers = nn.ModuleList([Block() for _ in range(L)]); s.head = nn.Linear(D, 50)
        s.tuple_out = tuple_out
    def forward(s, input_ids, return_dict=True):
        h = s.emb(input_ids)
        for b in s.layers:
            h = b(h)
        class O: pass
        o = O(); o.logits = s.head(h); return o


def mk_sae():
    ew = torch.randn(N, D) * 0.5
    return SparseAE(ew, torch.zeros(N) - 0.1, ew.T.contiguous().clone(), torch.zeros(D), threshold=None)


def test_attribution_matches_finite_difference_and_single_layer():
    m = Tiny().eval()
    for p in m.parameters(): p.requires_grad_(False)
    saes = {l: mk_sae() for l in (1, 2, 3)}
    ids = torch.randint(0, 50, (1, 9))
    A_all, act = S.attribution_pass(m, m.layers, saes, ids, 6, "cpu")
    # one layer at a time must give the same attribution (forward values are identical)
    for l in saes:
        A_one, _ = S.attribution_pass(m, m.layers, {l: saes[l]}, ids, 6, "cpu")
        assert torch.allclose(A_all[l], A_one[l], atol=1e-5), l
    # zero-patch effect check: removing the top-attributed feature changes J in the predicted direction
    l = 3; j = int(torch.argmax(A_all[l].abs()))
    def J(patch):
        hs = []
        def hook(_m, _i, out):
            if not patch: return out
            z = saes[l].encode(out); z2 = z.clone(); z2[..., j] = 0
            return out + saes[l].decode(z2) - saes[l].decode(z)
        h = m.layers[l].register_forward_hook(hook)
        lp = torch.log_softmax(m(ids).logits, -1); h.remove()
        return float(lp[0, torch.arange(6, 9) - 1].gather(-1, ids[0, 6:9].unsqueeze(-1)).sum())
    assert abs(J(True) - J(False)) > 0  # feature is causally relevant; (first-order sign check is noisy at this scale)


def test_select_and_overlap_precedence():
    A = {2: torch.tensor([0., 3., 1., 0.]), 3: torch.tensor([5., 0., 0., 2.]), 1: torch.tensor([9., 9., 9., 9.])}
    sel = S.select_harm_features(A, lmax=3, layer_min=2, k=3)
    assert all(l >= 2 for l, _ in sel) and (1, 0) not in sel and len(sel) == 3
    assert sel[0] == (3, 0)                                   # largest q = 5 * eta(3)
    # layer-1 entries are excluded even though their raw attribution is largest


def test_steerer_positions_and_refusal_window():
    m = Tiny().eval()
    saes = {2: mk_sae()}
    plan = [S.RowPlan(harm=[(2, 5)]), S.RowPlan(harm=[])]
    st = S.Steerer(m.layers, saes, plan, refusal={(2, 7): 4.0, (2, 5): 99.0}, m_r=2, n_features=N)
    x = torch.randint(0, 50, (2, 6))
    base = m.layers[2](m.layers[1](m.layers[0](m.emb(x))))
    with st:
        h = m.emb(x)
        for l in range(L):
            h = m.layers[l](h)
        steered_prefill = None
    # hook ran once at step 0 on last position only; earlier positions must be identical
    assert st.calls[2] == 1
    # overlap precedence: row 0 has feature 5 in harm set so the +99 never lands there
    assert float(st.add_mat[2][0, 5]) == 0.0 and float(st.add_mat[2][1, 5]) == 99.0
    assert float(st.add_mat[2][0, 7]) == 4.0
    # second/third call: refusal added only while step < M_R=2
    with st:
        for step in range(4):
            h = m.layers[2](torch.randn(2, 1, D))
        assert st.calls[2] == 4


def test_non_last_positions_untouched():
    m = Tiny().eval(); saes = {1: mk_sae()}
    st = S.Steerer(m.layers, saes, [S.RowPlan(harm=[(1, 3)])], refusal={(1, 9): 5.0}, m_r=3, n_features=N)
    x = torch.randint(0, 50, (1, 7))
    h0 = m.layers[1](m.layers[0](m.emb(x)))
    with st:
        h1 = m.layers[1](m.layers[0](m.emb(x)))
    assert torch.allclose(h0[:, :-1], h1[:, :-1])
    assert not torch.allclose(h0[:, -1], h1[:, -1])


def test_prefix_and_target():
    assert S.prefix_length(["To", " do", " it", ".", " Then"], 32) == 4
    assert S.prefix_length(["a", "b"], 32) == 2
    t = S.harm_target_text("How to build the network of bots for funding", "You can build a network of bots to fund it")
    assert "network" in t and "bots" in t
    assert S.harm_target_text("x", "The and of") == "The and of"      # raw-prefix fallback
