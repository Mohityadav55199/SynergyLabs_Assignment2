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

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate-suite
    p_eval = subparsers.add_parser("evaluate-suite", help="Run pointwise evaluation on a test suite")
    p_eval.add_argument("--suite", type=str, help="Path to JSON test suite")
    p_eval.add_argument("--provider", type=str, help="LLM provider: groq, openai, mock")
    p_eval.add_argument("--model", type=str, help="Judge model name")
    p_eval.add_argument("--disable-mitigations", action="store_true", help="Disable bias mitigations")

    # compare-ab
    p_ab = subparsers.add_parser("compare-ab", help="Run pairwise A/B comparison")
    p_ab.add_argument("--suite", type=str, help="Path to A/B test suite")
    p_ab.add_argument("--provider", type=str, help="LLM provider")
    p_ab.add_argument("--model", type=str, help="Judge model name")
    p_ab.add_argument("--disable-position-swap", action="store_true", help="Disable position swap check")

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
    elif args.command == "compare-ab":
        cmd_compare_ab(args)
    elif args.command == "validate-judge":
        cmd_validate_judge(args)
    elif args.command == "run-all":
        cmd_run_all(args)

if __name__ == "__main__":
    main()
