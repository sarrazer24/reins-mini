import os, sys, json, tempfile, random, importlib.util, argparse
ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location("ja", os.path.join(ROOT, "scripts", "16_judge_agreement.py"))
ja = importlib.util.module_from_spec(spec); spec.loader.exec_module(ja)

def V(h, ref=False): return dict(label=h, refusal=ref, empty_or_degenerate=False, semantic_coherent=True)

def test_kappa():
    assert ja.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    rng = random.Random(0); x = [rng.random() < .5 for _ in range(4000)]; y = [rng.random() < .5 for _ in range(4000)]
    assert abs(ja.kappa(x, y)) < 0.06

def test_main_pass_and_fail():
    with tempfile.TemporaryDirectory() as td:
        sd = os.path.join(td, "seed0"); os.makedirs(sd)
        rates = dict(original=.9, random_any=.9, random_active=.9, harm_only=.8, refusal_only=.4, reins=.2)
        rng = random.Random(1)
        for m, p in rates.items():
            for j, flip in (("A", 0.0), ("B", 0.05)):
                with open(os.path.join(sd, f"{m}.judged_{j}.jsonl"), "w") as f:
                    for i in range(300):
                        h = (i / 300) < p
                        if rng.random() < flip: h = not h
                        f.write(json.dumps(dict(sid=f"s{i}", method=m, verdict=V(h, not h))) + "\n")
        kp, st = ja.main(argparse.Namespace(runs=td, judge_a="A", judge_b="B", out=os.path.join(td, "o.csv")))
        assert kp > 0.8 and all(st["J3"])
