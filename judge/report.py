"""Aggregation: per-case verdicts -> suite report, pairwise results -> win rates, and the A/B winner rule."""
from typing import Dict, List, Optional

from . import stats
from .rubric import CRITERIA_IDS

MIN_DECISIVE_FOR_WINNER = 8
ALPHA = 0.05


def pointwise_summary(verdicts: List[dict]) -> dict:
    ok = [v for v in verdicts if v["status"] == "ok"]
    n = len(verdicts)
    passed = sum(1 for v in ok if v["passed"])
    overall = [v["overall"] for v in ok if v["overall"] is not None]
    per_crit = {cid: stats.mean([v["criteria"][cid]["score"] for v in ok if cid in v["criteria"] and v["criteria"][cid]["score"] is not None]) for cid in CRITERIA_IDS}
    return {
        "cases": n, "scored": len(ok), "unparseable": sum(1 for v in verdicts if v["status"] == "unparseable"), "api_errors": sum(1 for v in verdicts if v["status"] == "api_error"),
        "pass_rate": round(passed / len(ok), 4) if ok else None,
        "pass_rate_ci95": stats.wilson_ci(passed, len(ok)),
        "pass_rate_unparseable_as_fail": round(passed / n, 4) if n else None,
        "mean_overall": stats.mean(overall), "stdev_overall": stats.stdev(overall),
        "range_overall": [min(overall), max(overall)] if overall else None,
        "modal_band_share": stats.modal_band_share(overall),
        "distinct_rounded_scores": len({round(x) for x in overall}),
        "mean_per_criterion": per_crit,
        "ungrounded_capped_criteria": sum(v.get("ungrounded_capped", 0) for v in ok),
    }


def pairwise_summary(results: List[dict]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    n = len(results)
    c = {"A": 0, "B": 0, "TIE": 0}
    for r in ok:
        c[r["final"]] += 1
    single = [r["single_order"] for r in ok]
    return {
        "pairs": n, "scored": len(ok), "unparseable": n - len(ok),
        "wins_a": c["A"], "wins_b": c["B"], "ties": c["TIE"],
        "win_rate_a": round(c["A"] / len(ok), 4) if ok else None, "win_rate_b": round(c["B"] / len(ok), 4) if ok else None,
        "tie_rate": round(c["TIE"] / len(ok), 4) if ok else None,
        "order_disagreement_rate": round(sum(1 for r in ok if not r["agree"]) / len(ok), 4) if ok else None,   # any difference between the two orders
        "hard_flip_rate": round(sum(1 for r in ok if r["hard_flip"]) / len(ok), 4) if ok else None,           # A in one order, B in the other
        "single_order_wins_a": sum(1 for s in single if s == "A"), "single_order_wins_b": sum(1 for s in single if s == "B"), "single_order_ties": sum(1 for s in single if s == "TIE"),
    }


def declare_winner(wins_a: int, wins_b: int, ties: int, name_a: str = "A", name_b: str = "B") -> dict:
    """Winner rule. Decisive pairs = pairs where both orders agreed on a winner (disagreements and ties are not wins).
    Declare X the winner only if (decisive >= MIN_DECISIVE_FOR_WINNER) and (exact two-sided sign test p < ALPHA); otherwise 'no significant difference'
    and report the lean. The sign test ignores margin size and treats pairs as independent, and with a small suite it has little power."""
    decisive = wins_a + wins_b
    p = stats.sign_test_p(wins_a, wins_b)
    lean = name_a if wins_a > wins_b else name_b if wins_b > wins_a else None
    if decisive < MIN_DECISIVE_FOR_WINNER:
        verdict, winner = f"INCONCLUSIVE: only {decisive} decisive pairs (< {MIN_DECISIVE_FOR_WINNER})", None
    elif p is not None and p < ALPHA:
        winner = lean
        verdict = f"WINNER: {winner} ({max(wins_a, wins_b)}-{min(wins_a, wins_b)} on {decisive} decisive pairs, sign test p={p})"
    else:
        verdict, winner = f"NO SIGNIFICANT DIFFERENCE (p={p}); lean: {lean or 'none'}", None
    return {"winner": winner, "lean": lean, "decisive_pairs": decisive, "ties_or_disagreements": ties, "p_value": p, "verdict": verdict,
            "rule": f"decisive >= {MIN_DECISIVE_FOR_WINNER} and exact two-sided sign test p < {ALPHA}; ties/disagreements excluded",
            "caveat": "Small suite: the test has low power (a real but modest difference will usually read as 'no significant difference'), pairs come from one hand-written input set, "
                      "and the judge itself is imperfect (see judge validation), so a declared winner is evidence for this suite, not a general claim."}


def compare_configs(pw: List[dict], name_a: str, name_b: str, verdicts_a: Optional[List[dict]] = None, verdicts_b: Optional[List[dict]] = None) -> dict:
    s = pairwise_summary(pw)
    out = {"config_a": name_a, "config_b": name_b, "pairwise": s,
           "decision": declare_winner(s["wins_a"], s["wins_b"], s["ties"], name_a, name_b)}
    if verdicts_a is not None and verdicts_b is not None:
        sa, sb = pointwise_summary(verdicts_a), pointwise_summary(verdicts_b)
        by_b = {v["id"]: v for v in verdicts_b if v["status"] == "ok"}
        paired = [(v["overall"], by_b[v["id"]]["overall"]) for v in verdicts_a if v["status"] == "ok" and v["id"] in by_b]
        out["pointwise"] = {name_a: sa, name_b: sb,
                            "paired_n": len(paired),
                            "mean_diff_a_minus_b": stats.mean([x - y for x, y in paired]) if paired else None,
                            "mean_diff_ci95_bootstrap": stats.bootstrap_mean_diff_ci([x for x, _ in paired], [y for _, y in paired]) if paired else None}
    return out
