"""Bias measurement + judge validation experiments. Each writes judge/results/<name>.json from REAL judge calls.

  python -m judge.experiments gold adversarial position verbosity sycophancy ab retest
  python -m judge.experiments all --force

Every experiment compares the "baseline" judge prompt (no mitigations) with the "mitigated" one on the same data.
All calls are cached in the audit log, so a re-run only pays for what is missing.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import stats
from .config import JudgeConfig, PASS_THRESHOLD, check_cross_family
from .core import Judge
from .llm import LLMClient
from .report import pairwise_summary, pointwise_summary
from .run import load_suite, run_suite

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
VARIANTS = ("baseline", "mitigated")


def _save(name: str, obj: dict, client: LLMClient, cfg: JudgeConfig, suffix: str = "") -> dict:
    obj["meta"] = {"experiment": name, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "judge_model": cfg.judge_model,
                   "generator_model": cfg.generator_model, "family_check": check_cross_family(cfg.judge_model, cfg.generator_model, True),
                   "pass_threshold": PASS_THRESHOLD, "usage_this_process": client.usage_summary()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}{suffix}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return obj


def _cases(file):
    return json.loads((DATA / file).read_text(encoding="utf-8"))["cases"]   # internal datasets have their own schemas


# --------------------------------------------------------------------------------------------- judge validation: gold labels (+ score clustering)
def gold_metrics(cases, verdicts):
    by = {v["id"]: v for v in verdicts}
    scored = [c for c in cases if by[c["id"]]["status"] == "ok"]
    gold = [c["gold_pass"] for c in scored]
    pred = [by[c["id"]]["passed"] for c in scored]
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    tn = sum(1 for g, p in zip(gold, pred) if not g and not p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    k = stats.cohen_kappa(gold, pred)
    ov_pass = [by[c["id"]]["overall"] for c in scored if c["gold_pass"]]
    ov_fail = [by[c["id"]]["overall"] for c in scored if not c["gold_pass"]]
    mae = stats.mean([abs(by[c["id"]]["overall"] - c["gold_score"]) for c in scored])
    return {"n_total": len(cases), "n_scored": len(scored), "unparseable": len(cases) - len(scored),
            "agreement_rate": stats.agreement_rate(gold, pred), "agreement_ci95": stats.wilson_ci(tp + tn, len(scored)),
            "cohen_kappa": k, "kappa_band": stats.kappa_label(k),
            "confusion": {"true_pass_judged_pass": tp, "true_fail_judged_fail": tn, "true_fail_judged_pass(false_accept)": fp, "true_pass_judged_fail(false_reject)": fn},
            "mae_vs_gold_score": mae, "mean_overall_gold_pass": stats.mean(ov_pass), "mean_overall_gold_fail": stats.mean(ov_fail),
            "discrimination_gap": round(stats.mean(ov_pass) - stats.mean(ov_fail), 4) if ov_pass and ov_fail else None,
            "disagreements": [{"id": c["id"], "gold_pass": c["gold_pass"], "judge_overall": by[c["id"]]["overall"], "gold_note": c["gold_note"]} for c in scored if by[c["id"]]["passed"] != c["gold_pass"]]}


def exp_gold(client, cfg):
    cases = _cases("gold_cases.json")
    out = {"description": "Judge vs hand-labelled gold (26 cases, binary pass/fail). Baseline vs mitigated prompt; clustering = score spread.", "variants": {}}
    for var in VARIANTS:
        verdicts = [Judge(client, cfg, var).judge_case(c) for c in cases]
        s = pointwise_summary(verdicts)
        out["variants"][var] = {"validation": gold_metrics(cases, verdicts), "clustering": {k: s[k] for k in ("stdev_overall", "range_overall", "modal_band_share", "distinct_rounded_scores", "mean_overall")},
                                "summary": s, "verdicts": verdicts}
    return out


# --------------------------------------------------------------------------------------------- adversarial probes
def exp_adversarial(client, cfg):
    cases = _cases("adversarial_probes.json")
    out = {"description": "verbose-but-wrong (must FAIL) and terse-but-correct (must PASS), reference-free. fooled = verdict opposite to the truth.", "variants": {}}
    for var in VARIANTS:
        verdicts = [Judge(client, cfg, var).judge_case(c) for c in cases]
        by = {v["id"]: v for v in verdicts}
        rows, tally = [], {}
        for c in cases:
            v = by[c["id"]]
            fooled = None if v["status"] != "ok" else (v["passed"] != c["gold_pass"])
            rows.append({"id": c["id"], "kind": c["kind"], "gold_pass": c["gold_pass"], "judge_passed": v["passed"], "overall": v["overall"], "fooled": fooled, "status": v["status"]})
            t = tally.setdefault(c["kind"], {"n": 0, "fooled": 0, "unscored": 0})
            t["n"] += 1
            t["fooled"] += 1 if fooled else 0
            t["unscored"] += 1 if fooled is None else 0
        out["variants"][var] = {"by_kind": tally, "rows": rows, "verdicts": verdicts}
    return out


# --------------------------------------------------------------------------------------------- position bias
def exp_position(client, cfg):
    cases = _cases("position_pairs.json")
    kinds = {c["id"]: c["kind"] for c in cases}
    gold = {c["id"]: c["gold_winner"] for c in cases}
    out = {"description": "22 pairs with known winner (clear, closer, both-wrong, near-equal, identical). Every pair judged in BOTH orders.", "variants": {}}
    for var in VARIANTS:
        judge = Judge(client, cfg, var)
        res = [judge.judge_pair(c) for c in cases]
        ok = [r for r in res if r["status"] == "ok"]
        s = pairwise_summary(res)

        def acc(pick):
            return round(sum(1 for r in ok if pick(r) == gold[r["id"]]) / len(ok), 4) if ok else None
        ident = [r for r in ok if kinds[r["id"]] == "identical"]
        decisive_wrong = lambda pick: sum(1 for r in ok if pick(r) in ("A", "B") and gold[r["id"]] in ("A", "B") and pick(r) != gold[r["id"]])
        out["variants"][var] = {
            "summary": s,
            "single_order_AB": {"accuracy_vs_gold": acc(lambda r: r["orders"]["AB"]["winner"]), "picks_A_share": round(sum(1 for r in ok if r["orders"]["AB"]["winner_shown"] == "A") / len(ok), 4) if ok else None},
            "single_order_BA": {"accuracy_vs_gold": acc(lambda r: r["orders"]["BA"]["winner"]), "picks_first_shown_share": round(sum(1 for r in ok if r["orders"]["BA"]["winner_shown"] == "A") / len(ok), 4) if ok else None},
            "both_orders_require_agreement": {"accuracy_vs_gold": acc(lambda r: r["final"]), "tie_rate": s["tie_rate"], "decisive_but_wrong": decisive_wrong(lambda r: r["final"])},
            "identical_pairs": {"n": len(ident), "single_order_nonTIE": sum(1 for r in ident if r["orders"]["AB"]["winner"] != "TIE"), "both_orders_final_nonTIE": sum(1 for r in ident if r["final"] != "TIE")},
            "disagreeing_pairs": [{"id": r["id"], "kind": kinds[r["id"]], "AB": r["orders"]["AB"]["winner"], "BA": r["orders"]["BA"]["winner"], "gold": gold[r["id"]]} for r in ok if not r["agree"]],
            "pairs": res}
    return out


# --------------------------------------------------------------------------------------------- verbosity
def exp_verbosity(client, cfg):
    items = _cases("verbosity_pairs.json")
    pair_cases = [{"id": c["id"], "input": c["input"], "system_prompt": c["system_prompt"], "output_a": c["concise"], "output_b": c["padded"]} for c in items]
    out = {"description": "Same facts, concise (A) vs padded with filler and restatement (B). A length-blind judge should TIE or prefer the concise one.", "variants": {}}
    for var in VARIANTS:
        judge = Judge(client, cfg, var)
        res = [judge.judge_pair(c) for c in pair_cases]
        ok = [r for r in res if r["status"] == "ok"]
        singles = [r["orders"][o]["winner"] for r in ok for o in ("AB", "BA")]
        pw_c = [judge.judge_case({"id": c["id"] + "|concise", "input": c["input"], "system_prompt": c["system_prompt"], "model_output": c["concise"]}) for c in items]
        pw_p = [judge.judge_case({"id": c["id"] + "|padded", "input": c["input"], "system_prompt": c["system_prompt"], "model_output": c["padded"]}) for c in items]
        pairs = [(a["overall"], b["overall"]) for a, b in zip(pw_c, pw_p) if a["status"] == "ok" and b["status"] == "ok"]
        out["variants"][var] = {
            "pairwise_final": {"n": len(ok), "padded_wins": sum(1 for r in ok if r["final"] == "B"), "concise_wins": sum(1 for r in ok if r["final"] == "A"), "ties": sum(1 for r in ok if r["final"] == "TIE")},
            "pairwise_single_judgments": {"n": len(singles), "padded_preferred": singles.count("B"), "concise_preferred": singles.count("A"), "tie": singles.count("TIE"),
                                          "padded_preferred_rate": round(singles.count("B") / len(singles), 4) if singles else None},
            "pointwise": {"n": len(pairs), "mean_overall_concise": stats.mean([a for a, _ in pairs]), "mean_overall_padded": stats.mean([b for _, b in pairs]),
                          "mean_padded_minus_concise": stats.mean([b - a for a, b in pairs]), "padded_scored_higher": sum(1 for a, b in pairs if b > a), "padded_scored_lower": sum(1 for a, b in pairs if b < a)},
            "pairs": res, "pointwise_concise": pw_c, "pointwise_padded": pw_p}
    return out


# --------------------------------------------------------------------------------------------- sycophancy / style
def exp_sycophancy(client, cfg):
    cases = _cases("sycophancy_probes.json")
    out = {"description": "Confidently wrong answers (fake authority; half also carry a note addressed to the evaluator). All are wrong, so any PASS = fooled.", "variants": {}}
    for var in VARIANTS:
        verdicts = [Judge(client, cfg, var).judge_case(c) for c in cases]
        ok = [v for v in verdicts if v["status"] == "ok"]
        by_kind = {}
        for c, v in zip(cases, verdicts):
            t = by_kind.setdefault(c["kind"], {"n": 0, "fooled": 0, "unscored": 0, "overalls": []})
            t["n"] += 1
            if v["status"] != "ok":
                t["unscored"] += 1
            else:
                t["fooled"] += 1 if v["passed"] else 0
                t["overalls"].append(v["overall"])
        out["variants"][var] = {"n_scored": len(ok), "fooled": sum(1 for v in ok if v["passed"]), "fool_rate": round(sum(1 for v in ok if v["passed"]) / len(ok), 4) if ok else None,
                                "mean_overall": stats.mean([v["overall"] for v in ok]), "by_kind": by_kind, "ungrounded_capped_criteria": sum(v.get("ungrounded_capped", 0) for v in ok), "verdicts": verdicts}
    return out


# --------------------------------------------------------------------------------------------- A/B (prompt v1 vs v2)
def exp_ab(client, cfg):
    suite = load_suite(DATA / "ab_suite.json")
    rep = run_suite(suite, Judge(client, cfg, "mitigated"), "pairwise", also_pointwise=True)
    rep["generator_model"] = suite.get("generator_model")
    rep["lengths_chars"] = {"mean_output_a": stats.mean([len(c["output_a"]) for c in suite["cases"]]), "mean_output_b": stats.mean([len(c["output_b"]) for c in suite["cases"]])}
    return rep


# --------------------------------------------------------------------------------------------- test-retest
def _flip(v1, v2):
    pairs = [(a, b) for a, b in zip(v1, v2) if a["status"] == "ok" and b["status"] == "ok"]
    return {"n_compared": len(pairs), "pass_flips": sum(1 for a, b in pairs if a["passed"] != b["passed"]),
            "pass_flip_rate": round(sum(1 for a, b in pairs if a["passed"] != b["passed"]) / len(pairs), 4) if pairs else None,
            "mean_abs_score_diff": stats.mean([abs(a["overall"] - b["overall"]) for a, b in pairs]),
            "max_abs_score_diff": max((abs(a["overall"] - b["overall"]) for a, b in pairs), default=None),
            "flipped_ids": [a["id"] for a, b in pairs if a["passed"] != b["passed"]]}


def exp_retest(client, cfg):
    cases = _cases("gold_cases.json")[::2]   # every second gold case (13 of 26): the free-tier rate limit made the full set too slow
    run = lambda temp, tag: [Judge(client, cfg, "mitigated", temperature=temp).judge_case(c, run_tag=tag) for c in cases]
    first_t0, second_t0 = run(0.0, ""), run(0.0, "rerun")
    hot1, hot2 = run(0.7, "hot1"), run(0.7, "hot2")
    return {"description": "Mitigated judge on 13 of the 26 gold cases (every second one), repeated. Temperature 0: run vs rerun. Temperature 0.7: two independent runs.",
            "temperature_0": _flip(first_t0, second_t0), "temperature_0.7": _flip(hot1, hot2),
            "temperature_0_vs_0.7": _flip(first_t0, hot1)}


EXPERIMENTS = {"gold": exp_gold, "adversarial": exp_adversarial, "position": exp_position, "verbosity": exp_verbosity, "sycophancy": exp_sycophancy, "ab": exp_ab, "retest": exp_retest}


def main(argv=None) -> int:
    global VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="experiment names or 'all'")
    ap.add_argument("--force", action="store_true", help="recompute even if the results file exists")
    ap.add_argument("--judge-model", default=None, help="override JUDGE_MODEL for this run (results are stored with --suffix)")
    ap.add_argument("--suffix", default="", help="result-file suffix, e.g. _gptoss20b")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=None, help="restrict the baseline/mitigated comparison (saves calls)")
    a = ap.parse_args(argv)
    if a.variants:
        VARIANTS = tuple(a.variants)
    names = list(EXPERIMENTS) if a.names == ["all"] else a.names
    cfg = JudgeConfig()
    if a.judge_model:
        cfg.judge_model = a.judge_model
    client = LLMClient(cfg)
    for n in names:
        if n not in EXPERIMENTS:
            print(f"unknown experiment {n!r}; choose from {list(EXPERIMENTS)}", file=sys.stderr)
            return 2
        if (RESULTS / f"{n}{a.suffix}.json").exists() and not a.force:
            print(f"[skip] {n}: results exist (use --force)")
            continue
        print(f"[run] {n} ...", flush=True)
        _save(n, EXPERIMENTS[n](client, cfg), client, cfg, a.suffix)
        print(f"[done] {n} | usage so far: {json.dumps(client.usage_summary())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
