import pytest
from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)

def test_health_check_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "judge_model" in data

def test_api_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_dashboard_report_endpoint():
    res = client.get("/api/reports/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "suite_report" in data
    assert "ab_report" in data

def test_api_evaluate_pointwise():
    payload = {
        "input": "Explain the time complexity of binary search.",
        "model_output": "Binary search has O(log n) time complexity because it halves the search space at each step.",
        "expected_output": "O(log n)",
        "provider": "mock"
    }
    res = client.post("/api/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "overall_score" in data
    assert data["overall_score"] >= 1.0
    assert "criterion_verdicts" in data

def test_api_compare_pairwise():
    payload = {
        "input": "Explain the difference between TCP and UDP.",
        "output_a": "TCP is connection-oriented and reliable with flow control, while UDP is connectionless and lightweight.",
        "output_b": "Hello! Both TCP and UDP are internet protocols. They send packets across computers.",
        "provider": "mock"
    }
    res = client.post("/api/compare", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "final_winner" in data
    assert data["final_winner"] in ["A", "B", "TIE"]
    assert "average_score_a" in data
    assert "average_score_b" in data
