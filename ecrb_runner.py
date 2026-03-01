#!/usr/bin/env python3
"""
ecrb_runner.py — ECRB CLI Entry Point
──────────────────────────────────────
Beautiful terminal interface for running the Enterprise Codebase Regression
Benchmark.  Uses ``click`` for argument parsing and ``rich`` for display.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from core.evaluator import (
    EvaluationResult,
    StepResult,
    TaskConfig,
    run_progressive_saturation,
)
from core.llm_client import LLMClientFactory
from core.polysemy_analyzer import PolysemyAnalyzer
from core.context_injector import ContextInjector

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "model_config.yaml"

console = Console()


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="ECRB")
def cli() -> None:
    """
    ⍜  ECRB — Enterprise Codebase Regression Benchmark

    Evaluate LLMs on their resilience to Channel Capacity Saturation
    and Adversarial Polysemy in real-world software engineering tasks.
    """


@cli.command()
@click.option(
    "--model",
    required=True,
    help="Model name as defined in model_config.yaml (e.g. gpt-4o).",
)
@click.option(
    "--custom-repo",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to the target repository to use as noise source.",
)
@click.option(
    "--target-task",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the task JSON (e.g. custom_task/migration_rules.json).",
)
@click.option(
    "--config",
    "config_path",
    default=str(_DEFAULT_CONFIG),
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to model_config.yaml.",
    show_default=True,
)
@click.option(
    "--step-size",
    default=10_000,
    type=int,
    help="Token increment per saturation step.",
    show_default=True,
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="Hard ceiling on noise tokens (defaults to model context window).",
)
def evaluate(
    model: str,
    custom_repo: str,
    target_task: str,
    config_path: str,
    step_size: int,
    max_tokens: int | None,
) -> None:
    """Run the Progressive Saturation evaluation on a single model."""

    # ── Banner ───────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]⍜  ECRB — Progressive Saturation Benchmark[/]\n"
                f"[dim]Model:[/] [bold]{model}[/]   "
                f"[dim]Repo:[/] [bold]{custom_repo}[/]"
            ),
            box=box.DOUBLE_EDGE,
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()

    # ── Load dependencies ────────────────────────────────────────────────
    with console.status("[bold green]Loading configuration…"):
        task_config = TaskConfig.from_json(target_task)
        factory = LLMClientFactory(config_path)
        llm_client = factory.get_client(model)

    console.print(f"  [green]✓[/] Task loaded: [italic]{target_task}[/]")
    console.print(f"  [green]✓[/] Model: [bold]{llm_client.model_name}[/]")
    console.print()

    # ── Polysemy analysis ────────────────────────────────────────────────
    with console.status("[bold green]Analysing codebase polysemy…"):
        analyzer = PolysemyAnalyzer(
            repo_path=custom_repo,
            task_string=task_config.task_description,
        )

    injector = ContextInjector()

    # ── Step callback (live progress) ────────────────────────────────────
    step_table = Table(
        title="Saturation Steps",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold magenta",
    )
    step_table.add_column("Step", justify="center", style="cyan", width=6)
    step_table.add_column("Noise Tokens", justify="right", style="yellow")
    step_table.add_column("Prompt Tokens", justify="right", style="blue")
    step_table.add_column("Files", justify="center")
    step_table.add_column("SAS", justify="center")
    step_table.add_column("Result", justify="center")

    step_counter = {"n": 0}

    def on_step(step: StepResult) -> None:
        step_counter["n"] += 1
        sas_display = (
            "[bold green]1 ✓[/]" if step.sas == 1 else "[bold red]0 ✗[/]"
        )
        result = "[green]PASS[/]" if step.sas == 1 else "[red]FAIL[/]"
        step_table.add_row(
            str(step_counter["n"]),
            f"{step.noise_tokens:,}",
            f"{step.total_prompt_tokens:,}",
            str(step.files_included),
            sas_display,
            result,
        )
        if step.sas == 0 and step.violations:
            for v in step.violations:
                console.print(f"    [red]⚠ {v}[/]")

    # ── Run benchmark ────────────────────────────────────────────────────
    console.print("[bold]Starting progressive saturation…[/]\n")

    result: EvaluationResult = run_progressive_saturation(
        model_name=model,
        repo_path=custom_repo,
        task_config=task_config,
        llm_client=llm_client,
        polysemy_analyzer=analyzer,
        context_injector=injector,
        step_size=step_size,
        max_tokens=max_tokens,
        on_step=on_step,
    )

    # ── Display step table ───────────────────────────────────────────────
    console.print()
    console.print(step_table)
    console.print()

    # ── Final results table ──────────────────────────────────────────────
    results_table = Table(
        title="⍜ ECRB Results",
        box=box.HEAVY_EDGE,
        show_lines=True,
        title_style="bold white on blue",
        padding=(0, 2),
    )
    results_table.add_column("Model", style="bold cyan", justify="left")
    results_table.add_column(
        "Adversarial Polysemy Index (API)", justify="center", style="yellow"
    )
    results_table.add_column(
        "Attention Degradation Threshold (ADT)",
        justify="center",
        style="magenta",
    )
    results_table.add_column("Status", justify="center")

    adt_display = (
        f"{result.adt_threshold:,} tokens"
        if result.adt_threshold is not None
        else "[bold green]SURVIVED ALL LEVELS[/]"
    )
    status = (
        "[bold red]COLLAPSED[/]"
        if result.adt_threshold is not None
        else "[bold green]RESILIENT[/]"
    )

    results_table.add_row(
        result.model_name,
        f"{result.api_score:.4f}",
        adt_display,
        status,
    )

    console.print(results_table)
    console.print()

    # ── Exit code ────────────────────────────────────────────────────────
    if result.adt_threshold is not None:
        console.print(
            f"[bold red]⚠  Model collapsed at {result.adt_threshold:,} "
            f"noise tokens.[/]"
        )
        sys.exit(1)
    else:
        console.print(
            "[bold green]✓  Model survived all saturation levels.[/]"
        )
        sys.exit(0)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
