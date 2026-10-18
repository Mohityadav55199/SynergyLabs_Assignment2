import json

from fastapi.testclient import TestClient

import src.server as server
from judge.config import JudgeConfig
from judge.core import Judge
from judge.llm import ChatResult, LLMClient
from judge.rubric import CRITERIA_IDS

client = TestClient(server.app)


class Scripted(LLMClient):
    def __init__(self, replies, tmp_path):
        super().__init__(JudgeConfig(api_key="x"), log_path=tmp_path / "web.jsonl")
        self.replies = list(replies)

    def chat(self, messages, model, temperature=0.0, max_tokens=3000, tag="", role="judge"):
        return ChatResult(self.replies.pop(0), 10, 5)


def _pw(score, quote="Canberra."):
    return json.dumps({"criteria": {c: {"score": score, "rationale": "r", "evidence": quote} for c in CRITERIA_IDS}, "overall": {"score": score, "rationale": "r"}})


def _install(monkeypatch, tmp_path, replies):
    monkeypatch.setattr(server, "_judge", lambda api_key: Judge(Scripted(replies, tmp_path), JudgeConfig(api_key="x"), "mitigated"))


def test_health():
    for path in ("/health", "/api/health"):
        r = client.get(path)
        assert r.status_code == 200 and r.json()["status"] == "healthy" and "judge_model" in r.json()


def test_evaluate_ok(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, [_pw(5)])
    r = client.post("/api/evaluate", json={"input": "Capital of Australia?", "model_output": "Canberra."})
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is True and body["overall_score"] == 5.0 and len(body["criterion_verdicts"]) == 6 and body["execution_mode"] == "live"


def test_evaluate_unparseable_is_502_not_a_fake_score(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, ["not json", "still not json"])
    r = client.post("/api/evaluate", json={"input": "q", "model_output": "a"})
    assert r.status_code == 502 and "No score was invented" in r.json()["detail"]


def test_evaluate_validation():
    assert client.post("/api/evaluate", json={"input": "  ", "model_output": "a"}).status_code == 400
    assert client.post("/api/evaluate", json={"input": "q"}).status_code == 422


def test_compare_runs_both_orders(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, ['{"winner": "A", "confidence": "high", "rationale": "x"}', '{"winner": "B", "confidence": "high", "rationale": "y"}'])
    r = client.post("/api/compare", json={"input": "q", "output_a": "good", "output_b": "bad"})
    assert r.status_code == 200 and r.json()["final_winner"] == "A" and r.json()["flip_detected"] is False


def test_compare_position_flip_is_reported(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, ['{"winner": "A"}', '{"winner": "A"}'])
    r = client.post("/api/compare", json={"input": "q", "output_a": "x", "output_b": "y"})
    assert r.json()["final_winner"] == "TIE" and r.json()["flip_detected"] is True and r.json()["hard_flip"] is True


def test_dashboard_endpoint_shape():
    r = client.get("/api/reports/dashboard")
    assert r.status_code == 200 and "results" in r.json()
