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
