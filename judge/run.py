"""CLI: suite in -> structured report out.

  python -m judge.run --suite judge/data/gold_cases.json --out judge/results/gold_report.json          # pointwise
  python -m judge.run --suite judge/data/ab_suite.json    --out judge/results/ab_report.json --pointwise  # A/B (both orders)

Pointwise suite: {"cases": [{id, input, system_prompt, model_output, expected_output?, criteria?}]}  (JSON or YAML)
A/B suite      : {"config_a": {...}, "config_b": {...}, "cases": [{id, input, system_prompt?, output_a, output_b}]}
Add --replay to rebuild a report offline from the call log (no API calls).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import JudgeConfig, PASS_THRESHOLD, check_cross_family
from .core import Judge
from .llm import LLMClient
from .report import compare_configs, pairwise_summary, pointwise_summary
from .rubric import rubric_as_json


def load_suite(path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    if isinstance(data, list):
        data = {"name": p.stem, "cases": data}
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError(f"{path}: suite must contain a non-empty 'cases' list")
    seen = set()
    for i, c in enumerate(data["cases"]):
        if not isinstance(c, dict) or "id" not in c or "input" not in c:
            raise ValueError(f"{path}: case #{i} needs 'id' and 'input'")
        if c["id"] in seen:
            raise ValueError(f"{path}: duplicate case id {c['id']!r}")
        seen.add(c["id"])
        if not (("model_output" in c) or ("output_a" in c and "output_b" in c)):
            raise ValueError(f"{path}: case {c['id']!r} needs 'model_output' (pointwise) or 'output_a'+'output_b' (pairwise)")
    return data


def detect_mode(suite: dict) -> str:
    return "pairwise" if "output_a" in suite["cases"][0] else "pointwise"


def run_suite(suite: dict, judge: Judge, mode: str, also_pointwise: bool = False, run_tag: str = "") -> dict:
    report = {"suite": suite.get("name"), "mode": mode, "variant": judge.variant}
    if mode == "pointwise":
        verdicts = [judge.judge_case(c, run_tag) for c in suite["cases"]]
        report["summary"] = pointwise_summary(verdicts)
        report["cases"] = verdicts
        return report
    pw = [judge.judge_pair(c, run_tag) for c in suite["cases"]]
    ca, cb = suite.get("config_a", {}), suite.get("config_b", {})
    va = vb = None
    if also_pointwise:
        va = [judge.judge_case({"id": c["id"], "input": c["input"], "system_prompt": ca.get("system_prompt") or c.get("system_prompt"), "model_output": c["output_a"],
                                "expected_output": c.get("expected_output")}, run_tag) for c in suite["cases"]]
        vb = [judge.judge_case({"id": c["id"], "input": c["input"], "system_prompt": cb.get("system_prompt") or c.get("system_prompt"), "model_output": c["output_b"],
                                "expected_output": c.get("expected_output")}, run_tag) for c in suite["cases"]]
    report["comparison"] = compare_configs(pw, ca.get("name", "A"), cb.get("name", "B"), va, vb)
    report["pairs"] = pw
    if va is not None:
        report["pointwise_a"], report["pointwise_b"] = va, vb
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge pipeline: suite in -> report out")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["auto", "pointwise", "pairwise"], default="auto")
    ap.add_argument("--variant", choices=["baseline", "mitigated"], default="mitigated")
    ap.add_argument("--pointwise", action="store_true", help="for A/B suites: also score each config pointwise (pass rate / mean score)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--log", default=None, help="JSONL audit log (default judge/logs/judge_calls.jsonl)")
    ap.add_argument("--replay", action="store_true", help="answer only from the log; no API calls")
    ap.add_argument("--allow-same-family", action="store_true")
    ap.add_argument("--run-tag", default="")
    a = ap.parse_args(argv)

    cfg = JudgeConfig()
    if a.judge_model:
        cfg.judge_model = a.judge_model
    suite = load_suite(a.suite)
    mode = detect_mode(suite) if a.mode == "auto" else a.mode
    gen_model = suite.get("generator_model") or cfg.generator_model
    try:
        family = check_cross_family(cfg.judge_model, gen_model, a.allow_same_family)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    client = LLMClient(cfg, log_path=a.log, replay_only=a.replay)
    judge = Judge(client, cfg, a.variant, a.temperature)
    report = run_suite(suite, judge, mode, a.pointwise, a.run_tag)
    report["meta"] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "suite_file": str(a.suite), "judge_model": cfg.judge_model,
                      "generator_model": gen_model, "family_check": family, "temperature": judge.temperature, "pass_threshold": PASS_THRESHOLD,
                      "log": str(client.log_path), "replay": a.replay, "usage": client.usage_summary(), "rubric": rubric_as_json()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report.get("summary") or report["comparison"]["decision"], indent=2, ensure_ascii=False))
    print(f"report -> {out} | usage: {json.dumps(client.usage_summary())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
