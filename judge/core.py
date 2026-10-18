"""Judging: pointwise (reference-based or reference-free) and pairwise A-vs-B run in BOTH orders.

Failure policy: a response that cannot be parsed/validated is retried once (with a corrective message and a different cache tag);
if it still fails the case is recorded as `unparseable` (or `api_error`) with no score. Nothing is guessed or defaulted.
"""
from typing import Dict, List, Optional

from .config import JudgeConfig
from .llm import LLMClient
from .parsing import evidence_supported, parse_pairwise, parse_pointwise
from .rubric import GROUNDED_CRITERIA, overall_from_criteria, pairwise_messages, passes, pointwise_messages, split_criteria

RETRY_NUDGE = ("Your previous reply could not be used ({why}). Reply again with ONLY the single JSON object in the requested shape: "
               "no prose, no code fences, every required field present, scores as numbers.")


class Judge:
    def __init__(self, client: LLMClient, config: Optional[JudgeConfig] = None, variant: str = "mitigated", temperature: Optional[float] = None, enforce_grounding: Optional[bool] = None):
        assert variant in ("baseline", "mitigated")
        self.client = client
        self.config = config or client.config
        self.variant = variant
        self.temperature = self.config.judge_temperature if temperature is None else temperature
        self.enforce_grounding = (variant == "mitigated") if enforce_grounding is None else enforce_grounding

    # ------------------------------------------------------------------ shared call + one retry
    def _ask(self, messages: List[Dict[str, str]], tag: str, parser):
        """Returns (parsed, attempts[list of dict]). At most two attempts."""
        attempts, parsed = [], None
        for model in self.config.judge_chain:   # primary judge, then live fallback models; never a mock
            msgs = messages
            api_failed = False
            for n in (1, 2):
                res = self.client.chat(msgs, model, self.temperature, self.config.judge_max_tokens, tag=tag + ("" if n == 1 else "#retry"))
                if res.error:
                    attempts.append({"n": n, "model": model, "error": res.error, "tokens": 0})
                    parsed, api_failed = None, True
                    break   # the client already retried transient errors: move on to the next model
                parsed = parser(res.text)
                attempts.append({"n": n, "model": model, "raw_response": res.text, "tokens": res.prompt_tokens + res.completion_tokens, "cached": res.cached,
                                 "parse_ok": parsed.ok, "parse_error": parsed.error})
                if parsed.ok:
                    return parsed, attempts
                msgs = messages + [{"role": "assistant", "content": (res.text or "")[:1500]}, {"role": "user", "content": RETRY_NUDGE.format(why=parsed.error or "invalid")}]
            if not api_failed:
                break   # reachable but unparseable twice: do not shop around for a friendlier answer
        return parsed, attempts

    # ------------------------------------------------------------------ pointwise
    def judge_case(self, case: dict, run_tag: str = "") -> dict:
        ids, _ = split_criteria(case.get("criteria"))
        mode = "reference_based" if case.get("expected_output") else "reference_free"
        tag = f"pointwise|{self.variant}|{case['id']}|t{self.temperature}|{run_tag}"
        parsed, attempts = self._ask(pointwise_messages(case, self.variant), tag, lambda raw: parse_pointwise(raw, ids))
        used = attempts[-1].get("model") if attempts else None
        base = {"id": case["id"], "mode": mode, "variant": self.variant, "attempts": len(attempts), "tokens": sum(a.get("tokens", 0) for a in attempts),
                "judge_model_used": used, "fallback_used": bool(used and used != self.config.judge_model)}
        if parsed is None or not parsed.ok:
            last = attempts[-1]
            status = "api_error" if "error" in last else "unparseable"
            return {**base, "status": status, "error": last.get("error") or last.get("parse_error"), "criteria": None, "overall": None, "passed": None}
        crit, scores, ungrounded = {}, {}, 0
        for cid, c in parsed.data["criteria"].items():
            raw_score, score, grounded = c["score"], c["score"], None
            if self.enforce_grounding and cid in GROUNDED_CRITERIA and raw_score is not None and raw_score >= 4:
                grounded = evidence_supported(c["evidence"], case["model_output"])
                if not grounded:
                    score, ungrounded = 3.0, ungrounded + 1   # a 4/5 with no verbatim support is capped at 3
            crit[cid] = {"score": score, "raw_score": raw_score, "rationale": c["rationale"], "evidence": c["evidence"], "grounded": grounded}
            scores[cid] = score
        overall = overall_from_criteria(scores)
        return {**base, "status": "ok", "criteria": crit, "overall": overall, "overall_reported": parsed.data["overall_reported"],
                "overall_rationale": parsed.data["overall_rationale"], "passed": passes(scores, overall), "ungrounded_capped": ungrounded}

    # ------------------------------------------------------------------ pairwise, both orders
    def judge_order(self, case: dict, swapped: bool, run_tag: str = "") -> dict:
        shown = {"input": case["input"], "system_prompt": case.get("system_prompt"),
                 "output_a": case["output_b"] if swapped else case["output_a"], "output_b": case["output_a"] if swapped else case["output_b"]}
        tag = f"pairwise|{self.variant}|{case['id']}|{'BA' if swapped else 'AB'}|t{self.temperature}|{run_tag}"
        parsed, attempts = self._ask(pairwise_messages(shown, self.variant), tag, parse_pairwise)
        used = attempts[-1].get("model") if attempts else None
        rec = {"order": "BA" if swapped else "AB", "tokens": sum(a.get("tokens", 0) for a in attempts), "attempts": len(attempts),
               "judge_model_used": used, "fallback_used": bool(used and used != self.config.judge_model)}
        if parsed is None or not parsed.ok:
            last = attempts[-1]
            return {**rec, "status": "api_error" if "error" in last else "unparseable", "winner": None, "winner_shown": None, "error": last.get("error") or last.get("parse_error")}
        shown_w = parsed.data["winner"]
        winner = {"A": "B", "B": "A"}.get(shown_w, shown_w) if swapped else shown_w   # map back to the ORIGINAL labels
        return {**rec, "status": "ok", "winner": winner, "winner_shown": shown_w, "confidence": parsed.data["confidence"], "rationale": parsed.data["rationale"]}

    def judge_pair(self, case: dict, run_tag: str = "") -> dict:
        ab = self.judge_order(case, False, run_tag)
        ba = self.judge_order(case, True, run_tag)
        return combine_orders(case["id"], ab, ba)


def combine_orders(case_id: str, ab: dict, ba: dict) -> dict:
    """Position-bias mitigation: require agreement between the two orders. Disagreement -> TIE (flagged), unparseable if either failed.

    `single_order` is what a pipeline WITHOUT the mitigation would have reported (first order only).
    """
    out = {"id": case_id, "orders": {"AB": ab, "BA": ba}, "single_order": ab.get("winner"), "tokens": ab.get("tokens", 0) + ba.get("tokens", 0)}
    if ab["status"] != "ok" or ba["status"] != "ok":
        return {**out, "status": "unparseable", "final": None, "agree": None, "hard_flip": None}
    wa, wb = ab["winner"], ba["winner"]
    agree = wa == wb
    hard_flip = {wa, wb} == {"A", "B"}
    return {**out, "status": "ok", "agree": agree, "hard_flip": hard_flip, "final": wa if agree else "TIE"}
