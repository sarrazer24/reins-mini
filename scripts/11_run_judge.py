"""Judge stage. Reads runs/.../<method>.jsonl, writes <method>.judged_<judge-id>.jsonl (resumable).
Backends: --endpoint http://127.0.0.1:8000/v1/chat/completions --model NAME   (vLLM/OpenAI-compatible, UNTESTED)
          --hf-model Qwen/Qwen2.5-7B-Instruct                                 (transformers, slow, UNTESTED)"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reinsmini import data as D, pipeline as P, judge as J

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True); ap.add_argument("--guise-dir", required=True)
ap.add_argument("--judge-id", required=True)
ap.add_argument("--endpoint"); ap.add_argument("--model"); ap.add_argument("--hf-model")
ap.add_argument("--allow-cpu", action="store_true")
ap.add_argument("--json-prefill", action="store_true", help="force the reply to start with {\"label\": (HF backend)")
ap.add_argument("--limit", type=int, default=0, help="judge only the first N rows per file (smoke test)")
ap.add_argument("--only", default="", help="substring filter on file names")
a = ap.parse_args()
byid = {s.sid: s for s in D.load_guise(a.guise_dir)}
if a.endpoint:
    chat = J.http_chat(a.endpoint, a.model)
else:
    import torch, time
    if not torch.cuda.is_available() and not a.allow_cpu:
        sys.exit("NO GPU: set the Kaggle accelerator to GPU T4 x2 (a 7B judge on CPU never finishes). Aborting.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.hf_model)
    m = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=torch.float16, device_map="auto").eval()
    PREFILL = '{"label": ' if a.json_prefill else ""
    def chat(system, user):
        text = tk.apply_chat_template([dict(role="system", content=system), dict(role="user", content=user)],
                                      add_generation_prompt=True, tokenize=False) + PREFILL
        ids = tk(text, return_tensors="pt", add_special_tokens=False).to(m.device)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=100, do_sample=False)
        return PREFILL + tk.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
import time
T0 = time.time(); N = 0; DIAG = []
for f in sorted(glob.glob(os.path.join(a.run_dir, "*.jsonl"))):
    b = os.path.basename(f)
    if ".judged_" in b or b.startswith("plans") or (a.only and a.only not in b): continue
    out = f.replace(".jsonl", f".judged_{a.judge_id}.jsonl")
    done = P.done_ids(out)
    field = "safe" if b.endswith("_safe.jsonl") else "harmful"
    rows = [r for r in P.read_jsonl(f) if r["sid"] not in done]
    if a.limit: rows = rows[: a.limit]
    new = []
    for i, r in enumerate(rows):
        v = J.judge_one(chat, getattr(byid[r["sid"]], field), r["response"], diag=DIAG)
        new.append(dict(sid=r["sid"], method=r["method"], verdict=v)); N += 1
        if N % 5 == 0 or N <= 3:
            per = (time.time() - T0) / N; print(f"  {b} {i+1}/{len(rows)}  {per:.1f}s per judgment", flush=True)
            if N == 3 and per > 60: sys.exit(f"too slow ({per:.0f}s per judgment): check the accelerator / use a smaller judge")
        if len(new) >= 20 or i == len(rows) - 1:
            P.append_jsonl(out, new); new = []
    print(b, "judged", len(rows), "| failed replies:", len(DIAG))
    for d_ in DIAG[:3]: print("   unparseable reply (len %d, first 100 chars): %r" % (len(d_), d_[:100]))
    DIAG.clear()
