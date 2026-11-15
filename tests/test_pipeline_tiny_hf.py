"""End-to-end CPU smoke test with a randomly initialised tiny HF Llama (checks hooks under KV-cache
generate, left padding, batching, resume). Does NOT test Qwen3.5 or the real SAEs."""
import os, sys, tempfile, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from reinsmini import pipeline as P, steer as S, data as D
from reinsmini.sae import SparseAE

def tiny():
    words = ["<pad>", "<eos>", "<unk>"] + [chr(97 + i) for i in range(26)] + [".", "?", ":", "\n", "Answer", "Question"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    torch.manual_seed(0); m = LlamaForCausalLM(cfg).eval()
    return m, tok

def mk_sae(d=32, n=128):
    ew = torch.randn(n, d) * .3
    return SparseAE(ew, torch.zeros(n) - .05, ew.T.contiguous().clone(), torch.zeros(d))

def test_end_to_end():
    m, tok = tiny()
    layers = m.model.layers
    name, found = S.find_layer_list(m, 4); assert len(found) == 4
    saes = {l: mk_sae() for l in (1, 2, 3)}
    hp = dict(layers=4, harm_layer_min=1, K_H=3, K_R=2, M_R=2, sae_width=128)
    samples = [D.Sample(f"s{i}", " ".join("abcdefg"[j % 7] for j in range(i, i + 5)), None, "x") for i in range(5)]
    GEN = dict(P.GEN); P.GEN.update(max_new_tokens=6, repetition_penalty=1.0, no_repeat_ngram_size=0)
    with tempfile.TemporaryDirectory() as td:
        o, pl, r = (os.path.join(td, f) for f in ("o.jsonl", "p.jsonl", "r.jsonl"))
        P.run_original(m, tok, samples, o, bs=2, device="cpu", log=lambda *_: None)
        assert len(P.read_jsonl(o)) == 5
        P.run_original(m, tok, samples, o, bs=2, device="cpu", log=lambda *_: None)      # resume: no duplicates
        assert len(P.read_jsonl(o)) == 5
        P.run_plans(m, tok, saes, layers, samples, o, pl, hp, device="cpu", log=lambda *_: None)
        plans = P.read_jsonl(pl); assert len(plans) == 5 and all(len(p["harm"]) <= 3 for p in plans)
        refusal = {(2, 5): 3.0, (3, 9): 3.0}
        for meth in ("reins", "harm_only", "refusal_only", "random_any", "random_active"):
            out = os.path.join(td, meth + ".jsonl")
            P.run_steered(m, tok, saes, layers, samples, pl, out, meth, refusal, hp, bs=2, device="cpu", log=lambda *_: None)
            assert len(P.read_jsonl(out)) == 5, meth
        # a huge refusal push must change outputs vs original; a no-op plan must not
        big = {(2, 5): 500.0, (3, 9): 500.0}
        P.run_steered(m, tok, saes, layers, samples, pl, os.path.join(td, "big.jsonl"), "refusal_only", big, hp, bs=2, device="cpu", log=lambda *_: None)
        orig = {x["sid"]: x["response"] for x in P.read_jsonl(o)}
        bigr = {x["sid"]: x["response"] for x in P.read_jsonl(os.path.join(td, "big.jsonl"))}
        assert any(orig[k] != bigr[k] for k in orig)
        P.run_steered(m, tok, saes, layers, samples, pl, os.path.join(td, "noop.jsonl"), "refusal_only", {}, hp, bs=2, device="cpu", log=lambda *_: None)
        noop = {x["sid"]: x["response"] for x in P.read_jsonl(os.path.join(td, "noop.jsonl"))}
        assert noop == orig   # hooks with empty plan reproduce the original output exactly (batching-invariant)
    P.GEN.clear(); P.GEN.update(GEN)
