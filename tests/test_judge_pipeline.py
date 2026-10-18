import json

import pytest

from judge.config import JudgeConfig, check_cross_family, family_of
from judge.core import Judge, combine_orders
from judge.llm import ChatResult, LLMClient
from judge.report import declare_winner, pairwise_summary, pointwise_summary
from judge.rubric import CRITERIA_IDS, overall_from_criteria, passes
from judge.run import load_suite
from judge.stats import agreement_rate, bootstrap_mean_diff_ci, cohen_kappa, modal_band_share, sign_test_p, wilson_ci


class FakeClient(LLMClient):
    """Scripted responses; records the prompts it saw."""
    def __init__(self, replies, tmp_path):
        super().__init__(JudgeConfig(api_key="x"), log_path=tmp_path / "log.jsonl")
        self.replies, self.seen = list(replies), []

    def chat(self, messages, model, temperature=0.0, max_tokens=3000, tag="", role="judge"):
        self.seen.append((tag, messages))
        r = self.replies.pop(0)
        return r if isinstance(r, ChatResult) else ChatResult(r, 100, 50)


def pw(score, evidence="Canberra.", **override):
    crit = {c: {"score": score, "rationale": "r", "evidence": evidence} for c in CRITERIA_IDS}
    for k, v in override.items():
        crit[k]["score"] = v
    return json.dumps({"criteria": crit, "overall": {"score": score, "rationale": "r"}})


CASE = {"id": "c1", "input": "Capital of Australia?", "system_prompt": "Be brief.", "model_output": "Canberra."}


# ------------------------------------------------------------------ kappa / stats
def test_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    # hand-computed: observed agreement 0.8, chance agreement 0.5 -> (0.8 - 0.5) / 0.5 = 0.6
    assert cohen_kappa([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]) == pytest.approx(0.6, abs=1e-3)


def test_kappa_degenerate_and_empty():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    assert cohen_kappa([], []) is None
    assert agreement_rate([1, 0], [1, 1]) == 0.5


def test_sign_test_and_ci():
    assert sign_test_p(8, 0) == pytest.approx(2 / 256, abs=1e-4)
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(0, 0) is None
    lo, hi = wilson_ci(9, 10)
    assert 0.55 < lo < 0.65 and hi > 0.97
    ci = bootstrap_mean_diff_ci([1, 2, 3, 4], [0, 1, 2, 3])
    assert ci == (1.0, 1.0)
    assert modal_band_share([3.8, 3.9, 4.0, 4.1, 1.0]) == 0.4


# ------------------------------------------------------------------ rubric math
def test_weights_and_pass_rule():
    assert overall_from_criteria({c: 5 for c in CRITERIA_IDS}) == 5.0
    assert overall_from_criteria({"correctness": 4, "safety": None}) == 4.0   # null dropped, weights renormalised
    s = {c: 5 for c in CRITERIA_IDS}
    s["instruction_following"] = 2   # weighted mean is still high, but the floor must fail it
    assert overall_from_criteria(s) > 3.5 and not passes(s, overall_from_criteria(s))
    assert passes({c: 4 for c in CRITERIA_IDS}, 4.0)
    assert not passes({c: 3 for c in CRITERIA_IDS}, 3.0)


# ------------------------------------------------------------------ pointwise: parsing, retry, unparseable, grounding
def test_pointwise_ok(tmp_path):
    v = Judge(FakeClient([pw(5)], tmp_path)).judge_case(CASE)
    assert v["status"] == "ok" and v["passed"] and v["overall"] == 5.0 and v["attempts"] == 1


def test_pointwise_retries_once_then_succeeds(tmp_path):
    c = FakeClient(["I think it is great", pw(4, evidence="Canberra.")], tmp_path)
    v = Judge(c).judge_case(CASE)
    assert v["status"] == "ok" and v["attempts"] == 2
    assert c.seen[1][0].endswith("#retry") and "could not be used" in c.seen[1][1][-1]["content"]


def test_pointwise_unparseable_never_invents_a_score(tmp_path):
    v = Judge(FakeClient(["nope", "still nope"], tmp_path)).judge_case(CASE)
    assert v["status"] == "unparseable" and v["overall"] is None and v["passed"] is None and v["criteria"] is None


def test_pointwise_api_error_is_recorded(tmp_path):
    v = Judge(FakeClient([ChatResult("", error="HTTP 500"), ChatResult("", error="HTTP 500")], tmp_path)).judge_case(CASE)
    assert v["status"] == "api_error" and v["overall"] is None


def test_ungrounded_high_score_is_capped_in_mitigated_only(tmp_path):
    fake_quote = pw(5, evidence="a sentence the model never wrote")
    mit = Judge(FakeClient([fake_quote], tmp_path), variant="mitigated").judge_case(CASE)
    base = Judge(FakeClient([fake_quote], tmp_path), variant="baseline").judge_case(CASE)
    assert mit["criteria"]["correctness"]["score"] == 3.0 and mit["criteria"]["correctness"]["raw_score"] == 5.0 and mit["ungrounded_capped"] == 3
    assert not mit["passed"] or mit["overall"] < 5
    assert base["criteria"]["correctness"]["score"] == 5.0


def test_reference_mode_detected(tmp_path):
    v = Judge(FakeClient([pw(5)], tmp_path)).judge_case({**CASE, "expected_output": "Canberra"})
    assert v["mode"] == "reference_based"


# ------------------------------------------------------------------ pairwise: BOTH orders
def pair(w):
    return json.dumps({"winner": w, "confidence": "high", "rationale": "r"})


PAIR = {"id": "p1", "input": "q", "output_a": "good", "output_b": "bad"}


def test_both_orders_agree_maps_swapped_labels_back(tmp_path):
    # order AB: judge says A (=good). order BA: judge says B (=good, shown second). Consistent -> final A.
    c = FakeClient([pair("A"), pair("B")], tmp_path)
    r = Judge(c).judge_pair(PAIR)
    assert r["final"] == "A" and r["agree"] and not r["hard_flip"]
    shown_ba = c.seen[1][1][-1]["content"]
    assert shown_ba.index("RESPONSE A:\n<<<\nbad") < shown_ba.index("RESPONSE B:\n<<<\ngood")   # second call really swapped the texts


def test_pure_position_bias_is_flagged_and_demoted_to_tie(tmp_path):
    r = Judge(FakeClient([pair("A"), pair("A")], tmp_path)).judge_pair(PAIR)   # always "first shown" wins
    assert r["hard_flip"] and not r["agree"] and r["final"] == "TIE" and r["single_order"] == "A"


def test_partial_disagreement_is_not_a_hard_flip(tmp_path):
    r = Judge(FakeClient([pair("A"), pair("TIE")], tmp_path)).judge_pair(PAIR)
    assert not r["agree"] and not r["hard_flip"] and r["final"] == "TIE"


def test_pair_with_unparseable_order_is_unscored(tmp_path):
    r = Judge(FakeClient([pair("A"), "garbage", "garbage"], tmp_path)).judge_pair(PAIR)
    assert r["status"] == "unparseable" and r["final"] is None


def test_combine_orders_direct():
    ok = lambda w: {"status": "ok", "winner": w, "tokens": 1}
    assert combine_orders("x", ok("B"), ok("B"))["final"] == "B"
    assert combine_orders("x", ok("B"), ok("A"))["hard_flip"]


# ------------------------------------------------------------------ aggregation
def test_pointwise_summary_counts(tmp_path):
    vs = [Judge(FakeClient([pw(5)], tmp_path)).judge_case({**CASE, "id": "a"}),
          Judge(FakeClient([pw(2)], tmp_path)).judge_case({**CASE, "id": "b"}),
          Judge(FakeClient(["x", "x"], tmp_path)).judge_case({**CASE, "id": "c"})]
    s = pointwise_summary(vs)
    assert s["cases"] == 3 and s["scored"] == 2 and s["unparseable"] == 1
    assert s["pass_rate"] == 0.5 and s["pass_rate_unparseable_as_fail"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["mean_overall"] == 3.5


def test_pairwise_summary_and_winner_rule():
    ok = lambda f, agree=True, hard=False, single="A": {"status": "ok", "final": f, "agree": agree, "hard_flip": hard, "single_order": single}
    s = pairwise_summary([ok("A"), ok("A"), ok("B"), ok("TIE", False, True), {"status": "unparseable"}])
    assert (s["wins_a"], s["wins_b"], s["ties"], s["unparseable"]) == (2, 1, 1, 1) and s["hard_flip_rate"] == 0.25
    assert declare_winner(9, 1, 0, "x", "y")["winner"] == "x"
    assert declare_winner(6, 4, 0)["winner"] is None                      # not significant
    d = declare_winner(3, 0, 10)
    assert d["winner"] is None and "INCONCLUSIVE" in d["verdict"]         # too few decisive pairs
    assert "caveat" in d and "rule" in d


# ------------------------------------------------------------------ audit log / replay
def test_every_call_logged_and_replayable(tmp_path):
    log = tmp_path / "audit.jsonl"
    c1 = FakeClient([], tmp_path)
    c1.log_path = log
    # exercise the real LLMClient logging/caching path with a stubbed HTTP layer
    class Resp:
        status_code = 200
        headers = {}
        def json(self): return {"choices": [{"message": {"content": pw(5)}}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
    class Http:
        calls = 0
        def post(self, *a, **k):
            Http.calls += 1
            return Resp()
    real = LLMClient(JudgeConfig(api_key="k"), log_path=log)
    real._http = Http()
    v1 = Judge(real).judge_case(CASE)
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["messages"][-1]["role"] == "user" and rec["raw_response"].startswith("{") and rec["prompt_tokens"] == 11 and "api_key" not in json.dumps(rec).lower()
    assert real.usage_summary()["openai/gpt-oss-120b"]["total_tokens"] == 18
    replay = LLMClient(JudgeConfig(api_key=""), log_path=log, replay_only=True)
    v2 = Judge(replay).judge_case(CASE)
    assert v2["overall"] == v1["overall"] and Http.calls == 1 and replay.usage_summary()["openai/gpt-oss-120b"]["cached_calls"] == 1
    miss = Judge(replay).judge_case({**CASE, "id": "other"})
    assert miss["status"] == "api_error" and "replay_miss" in miss["error"]


# ------------------------------------------------------------------ config / suite loading
def test_family_guard():
    assert family_of("openai/gpt-oss-120b") == "openai" and family_of("qwen/qwen3.8-27b") == "qwen"
    assert check_cross_family("openai/gpt-oss-120b", "qwen/qwen3.8-27b")["cross_family"]
    with pytest.raises(ValueError):
        check_cross_family("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    assert not check_cross_family("openai/gpt-oss-120b", "openai/gpt-oss-20b", allow_same=True)["cross_family"]


def test_independent_env_config(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "judge-x")
    monkeypatch.setenv("GENERATOR_MODEL", "gen-y")
    c = JudgeConfig()
    assert (c.judge_model, c.generator_model) == ("judge-x", "gen-y")


def test_load_suite_json_yaml_and_validation(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"cases": [CASE]}), encoding="utf-8")
    (tmp_path / "s.yaml").write_text("name: y\ncases:\n  - id: a\n    input: hi\n    model_output: yo\n", encoding="utf-8")
    assert load_suite(tmp_path / "s.json")["cases"][0]["id"] == "c1"
    assert load_suite(tmp_path / "s.yaml")["name"] == "y"
    (tmp_path / "bad.json").write_text(json.dumps({"cases": [{"id": "a", "input": "q"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite(tmp_path / "bad.json")
    (tmp_path / "dup.json").write_text(json.dumps({"cases": [CASE, CASE]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite(tmp_path / "dup.json")
