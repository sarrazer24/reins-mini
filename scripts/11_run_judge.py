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
a = ap.parse_args()
byid = {s.sid: s for s in D.load_guise(a.guise_dir)}
if a.endpoint:
    chat = J.http_chat(a.endpoint, a.model)
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.hf_model)
    m = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=torch.float16, device_map="auto").eval()
    def chat(system, user):
        ids = tk.apply_chat_template([dict(role="system", content=system), dict(role="user", content=user)],
                                     add_generation_prompt=True, return_tensors="pt", return_dict=True).to(m.device)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=100, do_sample=False)
        return tk.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
for f in sorted(glob.glob(os.path.join(a.run_dir, "*.jsonl"))):
    b = os.path.basename(f)
    if ".judged_" in b or b.startswith("plans"): continue
    out = f.replace(".jsonl", f".judged_{a.judge_id}.jsonl")
    done = P.done_ids(out)
    field = "safe" if b.endswith("_safe.jsonl") else "harmful"
    rows = [r for r in P.read_jsonl(f) if r["sid"] not in done]
    new = []
    for i, r in enumerate(rows):
        v = J.judge_one(chat, getattr(byid[r["sid"]], field), r["response"])
        new.append(dict(sid=r["sid"], method=r["method"], verdict=v))
        if len(new) >= 20 or i == len(rows) - 1:
            P.append_jsonl(out, new); new = []
    print(b, "judged", len(rows))
