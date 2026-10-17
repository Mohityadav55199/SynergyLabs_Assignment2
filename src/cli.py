import sys
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import config
from .llm_clients.factory import get_llm_client
from .judge.pointwise import PointwiseJudge
from .judge.pairwise import PairwiseJudge
from .evaluation.validator import JudgeValidator
from .evaluation.adversarial_runner import AdversarialProbeRunner
from .visualization.dashboard import DashboardGenerator

console = Console()

def cmd_evaluate_suite(args):
    console.print(Panel.fit("[bold cyan]Running Pointwise Evaluation Suite[/bold cyan]"))
    suite_path = Path(args.suite) if args.suite else (config.data_dir / "test_suites" / "general_qa.json")
    client = get_llm_client(provider=args.provider, model=args.model)
    judge = PointwiseJudge(llm_client=client, apply_mitigations=not args.disable_mitigations)

    report = judge.evaluate_suite(suite_path)

    # Print summary table
    table = Table(title=f"Suite Evaluation Summary: {report.suite_name}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Total Test Cases", str(report.total_cases))
    table.add_row("Passed Cases", f"[green]{report.passed_cases}[/green]")
    table.add_row("Failed Cases", f"[red]{report.failed_cases}[/red]")
    table.add_row("Pass Rate", f"{report.pass_rate * 100:.1f}%")
    table.add_row("Mean Score (1.0 - 5.0)", f"{report.mean_overall_score:.2f}")
    table.add_row("Score Std Dev", f"{report.score_std_dev:.3f}")
    table.add_row("Total Tokens", f"{report.total_tokens:,}")
    table.add_row("Total Cost (USD)", f"${report.total_cost_usd:.6f}")
    table.add_row("Average Latency", f"{report.average_latency_ms:.1f} ms")

    console.print(table)

    # Save artifact
    out_path = config.artifacts_dir / "suite_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    console.print(f"[green]Saved suite report to {out_path}[/green]")
    return report.model_dump()

def cmd_compare_ab(args):
    console.print(Panel.fit("[bold cyan]Running Pairwise A/B Comparison (With Position Bias Check)[/bold cyan]"))
    suite_path = Path(args.suite) if args.suite else (config.data_dir / "test_suites" / "config_a_vs_b.json")
    client = get_llm_client(provider=args.provider, model=args.model)
    judge = PairwiseJudge(llm_client=client, enable_position_swap=not args.disable_position_swap)

    report = judge.evaluate_ab_suite(suite_path)

    table = Table(title="Pairwise A/B Comparison Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Pairs Evaluated", str(report.total_pairs))
    table.add_row("Wins for Config A", f"[green]{report.wins_a}[/green] ({report.win_rate_a * 100:.1f}%)")
    table.add_row("Wins for Config B", f"[blue]{report.wins_b}[/blue] ({report.win_rate_b * 100:.1f}%)")
    table.add_row("Ties", f"[yellow]{report.ties}[/yellow] ({report.tie_rate * 100:.1f}%)")
    table.add_row("Declared Winner", f"[bold green]Config {report.declared_winner}[/bold green]")
    table.add_row("Win Margin", f"{report.margin * 100:.1f}%")
    table.add_row("Position Flip Rate", f"[bold red]{report.position_flip_rate * 100:.1f}%[/bold red]")
    table.add_row("Swap Consistency Rate", f"{report.swap_consistency_rate * 100:.1f}%")
    table.add_row("Total Execution Cost", f"${report.total_cost_usd:.6f} USD")

    console.print(table)

    out_path = config.artifacts_dir / "ab_comparison_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    console.print(f"[green]Saved A/B report to {out_path}[/green]")
    return report.model_dump()

def cmd_validate_judge(args):
    console.print(Panel.fit("[bold cyan]Running Judge Validation & Adversarial Probes[/bold cyan]"))
    gold_path = config.data_dir / "test_suites" / "gold_labeled_benchmark.json"
    probes_path = config.data_dir / "adversarial_probes" / "adversarial_suite.json"

    client = get_llm_client(provider=args.provider, model=args.model)
    judge = PointwiseJudge(llm_client=client, apply_mitigations=not args.disable_mitigations)

    validator = JudgeValidator(judge)
    val_report = validator.validate_against_gold(gold_path)

    adv_runner = AdversarialProbeRunner(judge)
    adv_report = adv_runner.run_probes(probes_path)

    # Print Validation Table
    m = val_report["metrics"]
    table_val = Table(title="Judge Ground Truth Validation Metrics")
    table_val.add_column("Metric", style="cyan")
    table_val.add_column("Measured Value", style="magenta")
    table_val.add_column("Benchmark Target", style="green")

    table_val.add_row("Agreement Rate", f"{m['agreement_rate'] * 100:.1f}%", ">= 85.0%")
    table_val.add_row("Cohen's Kappa (kappa)", f"{m['cohens_kappa']:.3f}", ">= 0.700")
    table_val.add_row("Spearman Rank Corr (rho)", f"{m['spearman_correlation']:.3f}", ">= 0.800")
    table_val.add_row("Pearson Corr (r)", f"{m['pearson_correlation']:.3f}", ">= 0.800")
    table_val.add_row("Mean Absolute Error (MAE)", f"{m['mean_absolute_error']:.3f}", "<= 0.500")
    table_val.add_row("Test-Retest Consistency", f"{m['test_retest_consistency'] * 100:.1f}%", ">= 95.0%")

    console.print(table_val)

    # Print Adversarial Table
    table_adv = Table(title="Adversarial Probe Robustness")
    table_adv.add_column("Metric", style="cyan")
    table_adv.add_column("Value", style="magenta")

    table_adv.add_row("Total Probes", str(adv_report.total_probes))
    table_adv.add_row("Times Fooled", f"[bold red]{adv_report.total_fooled}[/bold red]")
    table_adv.add_row("Robustness Rate", f"[bold green]{adv_report.robustness_rate * 100:.1f}%[/bold green]")

    console.print(table_adv)

    # Save artifacts
    val_out = config.artifacts_dir / "judge_validation_report.json"
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    adv_out = config.artifacts_dir / "adversarial_report.json"
    with open(adv_out, "w", encoding="utf-8") as f:
        json.dump(adv_report.model_dump(), f, indent=2)

    console.print(f"[green]Saved validation reports to {val_out} and {adv_out}[/green]")
    return val_report, adv_report.model_dump()

def cmd_run_all(args):
    console.print(Panel.fit("[bold green]Executing Full Evaluation Benchmark & Generating Deliverables[/bold green]"))
    suite_report = cmd_evaluate_suite(args)
    ab_report = cmd_compare_ab(args)
    val_report, adv_report = cmd_validate_judge(args)

    dashboard_path = DashboardGenerator.generate_html(
        suite_report=suite_report,
        ab_report=ab_report,
        validation_report=val_report,
        adversarial_report=adv_report,
        output_path=config.artifacts_dir / "evaluation_dashboard.html"
    )
    console.print(f"[bold green][SUCCESS] Generated Interactive HTML Dashboard: {dashboard_path}[/bold green]")

def cmd_evaluate_case(args):
    console.print(Panel.fit("[bold cyan]Evaluating Single Model Output[/bold cyan]"))
    client = get_llm_client(provider=args.provider, model=args.model)
    judge = PointwiseJudge(llm_client=client, apply_mitigations=not args.disable_mitigations)

    from .schema import TestCase
    tc = TestCase(
        id=args.id or "cli-case-001",
        input=args.input,
        model_output=args.output,
        expected_output=args.expected,
        reference_mode=args.reference_mode or "auto"
    )
    verdict = judge.evaluate_case(tc)

    table = Table(title=f"Case Evaluation Verdict: {tc.id}")
    table.add_column("Criterion / Attribute", style="cyan")
    table.add_column("Score / Value", style="magenta")

    table.add_row("Overall Score", f"[bold green]{verdict.overall_score:.2f} / 5.0[/bold green]")
    table.add_row("Verdict", "[green]PASS[/green]" if verdict.passed else "[red]FAIL[/red]")
    table.add_row("Judge Model", str(verdict.judge_model))
    table.add_row("Latency", f"{verdict.latency_ms:.1f} ms")
    table.add_row("Cost", f"${verdict.cost_usd:.6f}")
    table.add_row("Overall Rationale", verdict.overall_rationale)

    for cv in verdict.criterion_verdicts:
        quote = f" (Quote: '{cv.evidence_quote[:40]}...')" if cv.evidence_quote else ""
        table.add_row(f"Criterion: {cv.criterion_id}", f"{cv.score:.1f} - {cv.rationale}{quote}")

    console.print(table)
    return verdict.model_dump()

def cmd_compare_pair(args):
    console.print(Panel.fit("[bold cyan]Comparing Pair (Candidate A vs Candidate B)[/bold cyan]"))
    client = get_llm_client(provider=args.provider, model=args.model)
    judge = PairwiseJudge(llm_client=client, enable_position_swap=not args.disable_position_swap)

    from .schema import PairwiseTestCase
    tc = PairwiseTestCase(
        id=args.id or "cli-pair-001",
        input=args.input,
        output_a=args.output_a,
        output_b=args.output_b
    )
    verdict = judge.evaluate_pair(tc)

    table = Table(title="Pairwise Verdict with Position Swap Check")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Final Winner", f"[bold green]{verdict.final_winner}[/bold green]")
    table.add_row("Position Flip Detected", "[bold red]YES[/bold red]" if verdict.flip_detected else "[green]NO (Agreement)[/green]")
    table.add_row("Average Score A", f"{verdict.average_score_a:.2f}")
    table.add_row("Average Score B", f"{verdict.average_score_b:.2f}")
    table.add_row("Forward Order Winner", verdict.forward_eval.winner)
    if verdict.reverse_eval:
        table.add_row("Reverse Order Winner", verdict.reverse_eval.winner)
    table.add_row("Rationale", verdict.synthesis_rationale)

    console.print(table)
    return verdict.model_dump()

def cmd_check_family(args):
    from .mitigations.self_enhancement_bias import SelfEnhancementMitigator
    judge_m = args.judge_model or config.judge_model
    gen_m = args.generator_model or config.generator_model
    res = SelfEnhancementMitigator.check_cross_family_compliance(judge_m, gen_m)

    table = Table(title="Model Family Self-Enhancement Check")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Judge Model", f"{res['judge_model']} (Family: {res['judge_family']})")
    table.add_row("Generator Model", f"{res['generator_model']} (Family: {res['generator_family']})")
    table.add_row("Cross-Family Compliant", "[green]YES[/green]" if res['is_cross_family'] else "[bold red]NO[/bold red]")
    table.add_row("Self-Enhancement Risk", f"[bold {('red' if res['self_enhancement_risk'] == 'HIGH' else 'green')}]{res['self_enhancement_risk']}[/]")
    table.add_row("Recommendation", res['recommendation'])

    console.print(table)
    return res

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate-suite
    p_eval = subparsers.add_parser("evaluate-suite", help="Run pointwise evaluation on a test suite (JSON or YAML)")
    p_eval.add_argument("--suite", type=str, help="Path to JSON or YAML test suite")
    p_eval.add_argument("--provider", type=str, help="LLM provider: groq, openai, mock")
    p_eval.add_argument("--model", type=str, help="Judge model name")
    p_eval.add_argument("--disable-mitigations", action="store_true", help="Disable bias mitigations")

    # evaluate-case
    p_case = subparsers.add_parser("evaluate-case", help="Evaluate a single prompt and output")
    p_case.add_argument("--id", type=str, help="Case ID")
    p_case.add_argument("--input", type=str, required=True, help="Input prompt")
    p_case.add_argument("--output", type=str, required=True, help="Model output to evaluate")
    p_case.add_argument("--expected", type=str, help="Reference / Expected output")
    p_case.add_argument("--reference-mode", type=str, choices=["auto", "reference_based", "reference_free"], default="auto")
    p_case.add_argument("--provider", type=str, help="LLM provider: groq, openai, mock")
    p_case.add_argument("--model", type=str, help="Judge model name")
    p_case.add_argument("--disable-mitigations", action="store_true", help="Disable bias mitigations")

    # compare-ab
    p_ab = subparsers.add_parser("compare-ab", help="Run pairwise A/B comparison on a test suite (JSON or YAML)")
    p_ab.add_argument("--suite", type=str, help="Path to A/B test suite (JSON or YAML)")
    p_ab.add_argument("--provider", type=str, help="LLM provider")
    p_ab.add_argument("--model", type=str, help="Judge model name")
    p_ab.add_argument("--disable-position-swap", action="store_true", help="Disable position swap check")

    # compare-pair
    p_pair = subparsers.add_parser("compare-pair", help="Compare a single pair Candidate A vs Candidate B")
    p_pair.add_argument("--id", type=str, help="Pair ID")
    p_pair.add_argument("--input", type=str, required=True, help="Input prompt")
    p_pair.add_argument("--output-a", type=str, required=True, help="Model Output A")
    p_pair.add_argument("--output-b", type=str, required=True, help="Model Output B")
    p_pair.add_argument("--provider", type=str, help="LLM provider")
    p_pair.add_argument("--model", type=str, help="Judge model name")
    p_pair.add_argument("--disable-position-swap", action="store_true", help="Disable position swap")

    # check-family
    p_fam = subparsers.add_parser("check-family", help="Check self-enhancement bias risk between judge and generator")
    p_fam.add_argument("--judge-model", type=str, help="Judge model name")
    p_fam.add_argument("--generator-model", type=str, help="Generator model name")

    # validate-judge
    p_val = subparsers.add_parser("validate-judge", help="Validate judge against gold benchmark & adversarial probes")
    p_val.add_argument("--provider", type=str, help="LLM provider")
    p_val.add_argument("--model", type=str, help="Judge model name")
    p_val.add_argument("--disable-mitigations", action="store_true", help="Disable bias mitigations")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Execute all steps and generate deliverables & dashboard")
    p_all.add_argument("--suite", type=str, help="Path to test suite")
    p_all.add_argument("--provider", type=str, help="LLM provider")
    p_all.add_argument("--model", type=str, help="Judge model name")
    p_all.add_argument("--disable-mitigations", action="store_true", help="Disable bias mitigations")
    p_all.add_argument("--disable-position-swap", action="store_true", help="Disable position swap")

    args = parser.parse_args()

    if args.command == "evaluate-suite":
        cmd_evaluate_suite(args)
    elif args.command == "evaluate-case":
        cmd_evaluate_case(args)
    elif args.command == "compare-ab":
        cmd_compare_ab(args)
    elif args.command == "compare-pair":
        cmd_compare_pair(args)
    elif args.command == "check-family":
        cmd_check_family(args)
    elif args.command == "validate-judge":
        cmd_validate_judge(args)
    elif args.command == "run-all":
        cmd_run_all(args)

if __name__ == "__main__":
    main()
