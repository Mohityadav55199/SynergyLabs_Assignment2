"""Builds an A/B suite by running the GENERATOR model under two system prompts (prompt v1 vs v2).

  python -m judge.generate --inputs judge/data/ab_inputs.json --out judge/data/ab_suite.json
"""
import argparse
import json
import sys
from pathlib import Path

from .config import JudgeConfig
from .llm import LLMClient


def generate_suite(inputs_doc: dict, client: LLMClient, cfg: JudgeConfig, temperature: float = 0.0, max_tokens: int = 520) -> dict:
    cfgs = inputs_doc["configs"]
    cases, failed = [], []
    for i, q in enumerate(inputs_doc["inputs"], start=1):
        outs = {}
        for key in ("A", "B"):
            msgs = [{"role": "system", "content": cfgs[key]["system_prompt"]}, {"role": "user", "content": q}]
            res = client.chat(msgs, cfg.generator_model, temperature, max_tokens, tag=f"gen2|{key}|q{i:02d}", role="generator")
            cut = bool(max_tokens) and res.completion_tokens >= max_tokens   # cut off by the token cap: not a real answer, exclude the case
            outs[key] = None if (res.error or cut or not res.text.strip()) else res.text.strip()
        if outs["A"] is None or outs["B"] is None:
            failed.append(f"q{i:02d}")
            continue
        cases.append({"id": f"q{i:02d}", "input": q, "output_a": outs["A"], "output_b": outs["B"]})
    return {"name": "ab_prompt_v1_vs_v2", "generator_model": cfg.generator_model, "config_a": cfgs["A"], "config_b": cfgs["B"],
            "generation_temperature": temperature, "failed_generations": failed, "cases": cases}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    cfg = JudgeConfig()
    client = LLMClient(cfg, log_path=a.log)
    suite = generate_suite(json.loads(Path(a.inputs).read_text(encoding="utf-8")), client, cfg)
    suite["usage"] = client.usage_summary()
    Path(a.out).write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}: {len(suite['cases'])} cases, failed={suite['failed_generations']}, usage={json.dumps(client.usage_summary())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
