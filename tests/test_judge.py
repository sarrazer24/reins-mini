import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reinsmini import judge as J

def test_parse():
    assert J.parse_verdict('x {"label": 1, "refusal": false, "empty_or_degenerate": false, "semantic_coherent": true} y')["label"] is True
    assert J.parse_verdict("no json") is None
    assert J.parse_verdict('{"label": 1}') is None
    v = J.parse_verdict('{"label":0,"refusal":"true","empty_or_degenerate":true,"semantic_coherent":true}')
    assert v["semantic_coherent"] is False

def test_retry_and_drop():
    calls = {"n": 0}
    def flaky(s, u):
        calls["n"] += 1
        return "garbage" if calls["n"] < 3 else '{"label":0,"refusal":true,"empty_or_degenerate":false,"semantic_coherent":true}'
    assert J.judge_one(flaky, "p", "r")["refusal"] is True and calls["n"] == 3
    assert J.judge_one(lambda s, u: "garbage", "p", "r") is None
    assert J.judge_one(lambda s, u: 1/0, "p", "r", retries=0) is None
    assert J.judge_one(lambda s, u: "x", "p", "  ")["empty_or_degenerate"] is True

def test_diag_collects_failed_replies():
    d = []
    assert J.judge_one(lambda s, u: "oops no json", "p", "r", retries=1, diag=d) is None and d == ["oops no json", "oops no json"]

def test_typo_key_is_repaired_not_dropped():
    v = J.parse_verdict('{"label": 0, "refusal": true, "empty_or_degenerat": false, "semantic_coherent": true}')
    assert v == dict(label=False, refusal=True, empty_or_degenerate=False, semantic_coherent=True, key_repair=True)
    ok = J.parse_verdict('{"label": 1, "refusal": false, "empty_or_degenerate": false, "semantic_coherent": true}')
    assert "key_repair" not in ok                                   # exact keys (judge A) are untouched
    assert J.parse_verdict('{"label": 1, "refusal": false}') is None  # missing fields still drop
