import pytest
from src.parser import RobustJsonParser

def test_clean_markdown_fences():
    raw = "```json\n{\"overall_score\": 4.5, \"rationale\": \"Great\"}\n```"
    result = RobustJsonParser.parse_to_dict(raw)
    assert result["overall_score"] == 4.5
    assert result["rationale"] == "Great"

def test_trailing_commas_repair():
    raw = '{"overall_score": 3.8, "rationale": "Good",}'
    result = RobustJsonParser.parse_to_dict(raw)
    assert result["overall_score"] == 3.8

def test_conversational_preamble_postamble():
    raw = 'Here is the requested verdict:\n{"overall_score": 4.0, "winner": "A"}\nHope this helps!'
    result = RobustJsonParser.parse_to_dict(raw)
    assert result["overall_score"] == 4.0
    assert result["winner"] == "A"

def test_heuristic_regex_fallback():
    raw = "The score awarded is overall_score: 2.5 and overall_rationale: 'Severe logical gaps'."
    result = RobustJsonParser.heuristic_regex_extractor(raw)
    assert result["overall_score"] == 2.5
