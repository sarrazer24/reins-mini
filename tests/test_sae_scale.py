import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reinsmini.sae import SparseAE, scale_scan, apply_scales
from reinsmini import steer as S
import json, tempfile

def ident(d=8, thr=0.5):
    I = torch.eye(d)
    return SparseAE(torch.cat([I, -I]), torch.zeros(2 * d), torch.cat([I, -I], 1), torch.zeros(d), threshold=thr)

def test_scan_monotone_and_scale_roundtrip():
    torch.manual_seed(0); x = torch.randn(200, 8)
    sae = ident()
    r = scale_scan(x, sae, [0.25, 1.0, 4.0, 16.0])
    nm = [t[1] for t in r]; l0 = [t[2] for t in r]
    assert nm == sorted(nm, reverse=True) and l0 == sorted(l0)       # more scale: better reconstruction, more active features
    assert sae.input_scale is None                                    # scan restores the setting
    # recon is in x space: with a large scale and tiny threshold it reproduces x
    sae2 = ident(thr=0.0); sae2.input_scale = 5.0
    assert torch.allclose(sae2.decode(sae2.encode(x)), x, atol=1e-5)

def test_apply_scales():
    sae = ident(); saes = {3: sae, 4: ident()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump({"3": 2.5}, f)
    apply_scales(saes, f.name); assert saes[3].input_scale == 2.5 and saes[4].input_scale is None

def test_steerer_delta_is_in_x_space_with_scale():
    D_, N = 8, 16
    torch.manual_seed(1); ew = torch.randn(N, D_) * .3
    sae = SparseAE(ew, torch.zeros(N), ew.T.contiguous().clone(), torch.zeros(D_), None, input_scale=4.0)
    class B(torch.nn.Module):
        def forward(s, h): return h
    layers = torch.nn.ModuleList([B()])
    x = torch.randn(1, 3, D_)
    st = S.Steerer(layers, {0: sae}, [S.RowPlan()], {(0, 2): 6.0}, m_r=1, n_features=N)
    with st: y = layers[0](x)
    expected = 6.0 * sae.dec_w[:, 2] / 4.0                      # b * decoder column / input_scale
    assert torch.allclose(y[0, -1] - x[0, -1], expected, atol=1e-5)

def test_make_scales_from_real_scan():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "..", "scripts", "07_make_scales.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sc = m.scales_from_scan(os.path.join(os.path.dirname(__file__), "..", "results", "sae_scale_scan_fine.csv"))
    assert set(sc) == {str(l) for l in range(4, 24)}
    vals = [sc[str(l)] for l in range(4, 24)]
    assert sc["23"] < sc["20"] < sc["17"] < sc["14"] < sc["10"]        # coarse decrease with depth (small bumps exist at layers 5-12)
    assert 0.8 < sc["23"] < 1.1 and 3.5 < sc["14"] < 4.2
