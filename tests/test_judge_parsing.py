import json

from judge.parsing import evidence_supported, extract_json, parse_pairwise, parse_pointwise
from judge.rubric import CRITERIA_IDS


def good_pointwise(score=5, evidence="Canberra."):
    return json.dumps({"criteria": {c: {"score": score, "rationale": "ok", "evidence": evidence} for c in CRITERIA_IDS},
                       "overall": {"score": score, "rationale": "fine"}})


def test_plain_json():
    assert parse_pointwise(good_pointwise()).ok


def test_code_fence_and_prose():
    raw = "Sure! Here is my verdict:\n```json\n" + good_pointwise(4) + "\n```\nHope that helps."
    p = parse_pointwise(raw)
    assert p.ok and p.data["criteria"]["correctness"]["score"] == 4


def test_think_block_with_braces_is_stripped():
    raw = '<think>maybe {"criteria": 1} hmm</think>' + good_pointwise(3)
    assert parse_pointwise(raw).data["criteria"]["tone"]["score"] == 3


def test_unclosed_think_block_yields_no_json():
    assert not parse_pointwise('<think>I am still thinking {"a": 1}').ok


def test_trailing_comma_and_smart_quotes_repaired():
    raw = '{"winner": “A”, "confidence": "high", "rationale": "x",}'
    assert parse_pairwise(raw).data["winner"] == "A"


def test_outermost_object_wins_over_nested():
    obj = extract_json('noise {"criteria": {"correctness": {"score": 5}}, "overall": {"score": 5}} tail')
    assert "criteria" in obj and "overall" in obj


def test_braces_inside_strings_do_not_break_balancing():
    obj = extract_json('{"winner": "TIE", "rationale": "uses } and { in text"}')
    assert obj["winner"] == "TIE"


def test_missing_criterion_is_rejected_not_invented():
    d = json.loads(good_pointwise())
    del d["criteria"]["safety"]
    p = parse_pointwise(json.dumps(d))
    assert not p.ok and "safety" in p.error


def test_out_of_range_and_non_numeric_scores_rejected():
    assert not parse_pointwise(good_pointwise(7)).ok
    d = json.loads(good_pointwise())
    d["criteria"]["tone"]["score"] = "great"
    assert not parse_pointwise(json.dumps(d)).ok


def test_null_score_means_not_applicable():
    d = json.loads(good_pointwise())
    d["criteria"]["safety"]["score"] = None
    p = parse_pointwise(json.dumps(d))
    assert p.ok and p.data["criteria"]["safety"]["score"] is None


def test_empty_and_garbage():
    assert parse_pointwise("").error == "empty_response"
    assert parse_pointwise("I cannot judge this.").error == "no_json_found"
    assert not parse_pairwise('{"winner": "C"}').ok


def test_pairwise_label_normalisation():
    assert parse_pairwise('{"winner": "Response B"}').data["winner"] == "B"
    assert parse_pairwise('{"winner": "tie"}').data["winner"] == "TIE"


def test_evidence_must_be_verbatim():
    out = "A hash map stores key-value pairs.\nLookups are O(1)."
    assert evidence_supported("stores key-value pairs", out)
    assert evidence_supported("“stores key-value pairs” ... Lookups are O(1)", out)
    assert not evidence_supported("stores values in a tree", out)
    assert not evidence_supported(None, out)
    assert not evidence_supported("", out)
