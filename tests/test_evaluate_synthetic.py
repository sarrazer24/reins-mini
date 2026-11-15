"""Builds synthetic judged files with known outcomes and checks 12_evaluate.py reports what it should."""
import os, sys, json, subprocess, tempfile, csv
ROOT = os.path.join(os.path.dirname(__file__), "..")

def V(label=False, refusal=False, deg=False, coh=True): return dict(label=label, refusal=refusal, empty_or_degenerate=deg, semantic_coherent=coh)

def write(d, name, verdicts):
    with open(os.path.join(d, name), "w") as f:
        for i, v in enumerate(verdicts): f.write(json.dumps(dict(sid=f"s{i}", method="x", verdict=v)) + "\n")

def test_eval():
    with tempfile.TemporaryDirectory() as td:
        for seed in (0, 1):
            d = os.path.join(td, f"seed{seed}"); os.makedirs(d)
            n = 100
            write(d, "original.judged_t.jsonl", [V(True)] * 90 + [V()] * 10)
            write(d, "random_any.judged_t.jsonl", [V(True)] * 88 + [V()] * 12)
            write(d, "random_active.judged_t.jsonl", [V(True)] * 85 + [V()] * 15)
            write(d, "harm_only.judged_t.jsonl", [V(True)] * 40 + [V()] * 55 + [V(refusal=True)] * 5)
            write(d, "refusal_only.judged_t.jsonl", [V(True)] * 80 + [V(refusal=True)] * 20)
            write(d, "reins.judged_t.jsonl", [V(True)] * 25 + [V(refusal=True)] * 65 + [V()] * 7 + [V(deg=True, coh=False)] * 3)
            write(d, "original_safe.judged_t.jsonl", [V()] * 95 + [V(refusal=True)] * 5)
            write(d, "reins_safe.judged_t.jsonl", [V()] * 70 + [V(refusal=True)] * 30)
            with open(os.path.join(td, f"gate_scores_seed{seed}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["sid", "kind", "open"]); w.writeheader()
                for i in range(n):
                    w.writerow(dict(sid=f"s{i}", kind="harmful", open=int(i < 98)))
                    w.writerow(dict(sid=f"s{i}", kind="safe", open=int(i < 5)))
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "12_evaluate.py"), "--runs", td, "--judge-id", "t",
                              "--out", os.path.join(td, "res", "summary.csv")], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        txt = out.stdout
        assert "[PASS] C1" in txt and "[PASS] C2" in txt and "[PASS] C3" in txt and "[PASS] C4" in txt, txt
        assert "[PASS] C5 Gate" in txt and "[PASS] C5b" in txt and "[PASS] C6" in txt and "[PASS] C7a" in txt and "[PASS] C7b" in txt, txt
