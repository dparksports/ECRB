"""
core/evaluator.py — Progressive Saturation Loop
────────────────────────────────────────────────
Orchestrates the full ECRB benchmark by incrementally flooding the LLM's
context window and testing structural adherence at each noise level.

Architectural constraint  [⍜ STRICT_ISOLATION]:
  • Every dependency (LLM client, analyzer, injector, weaver) is received
    via explicit function arguments — zero global state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.context_injector import ContextInjector, InjectedPrompt
from core.llm_client import LLMClient
from core.polysemy_analyzer import AnalysisResult, PolysemyAnalyzer
from core.weaver import WeaverResult, calculate_sas


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepResult:
    """Result of a single noise-level iteration."""

    noise_tokens: int
    sas: int
    violations: List[str]
    files_included: int
    total_prompt_tokens: int


@dataclass
class EvaluationResult:
    """Final benchmark result for a model.

    Attributes
    ----------
    model_name:
        The evaluated model's identifier.
    api_score:
        Adversarial Polysemy Index of the target repo.
    adt_threshold:
        The token count at which the model's SAS first dropped to 0.
        ``None`` if the model survived all tested levels.
    history:
        Ordered list of per-step results.
    """

    model_name: str
    api_score: float
    adt_threshold: Optional[int]
    history: List[StepResult] = field(default_factory=list)


# ── Task Config Loader ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskConfig:
    """Parsed task configuration."""

    task_description: str
    constraint_prompt: str
    forbidden_imports: List[str]
    forbidden_globals: List[str]

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)

        weaver_rules = data.get("weaver_rules", {})
        return cls(
            task_description=data["task_description"],
            constraint_prompt=data["constraint_prompt"],
            forbidden_imports=weaver_rules.get("forbidden_imports", []),
            forbidden_globals=weaver_rules.get("forbidden_globals", []),
        )


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_progressive_saturation(
    *,
    model_name: str,
    repo_path: str | Path,
    task_config: TaskConfig,
    llm_client: LLMClient,
    polysemy_analyzer: PolysemyAnalyzer,
    context_injector: ContextInjector,
    step_size: int = 10_000,
    max_tokens: Optional[int] = None,
    on_step: Optional[Callable[[StepResult], None]] = None,
) -> EvaluationResult:
    """Run the progressive saturation benchmark.

    Parameters
    ----------
    model_name:
        Identifier for the model under test.
    repo_path:
        Path to the target codebase.
    task_config:
        Parsed task with description, constraint, and weaver rules.
    llm_client:
        Pre-configured LLM client.
    polysemy_analyzer:
        Analyzer instance pointed at *repo_path*.
    context_injector:
        Injector instance for prompt assembly.
    step_size:
        Token increment per iteration (default 10 000).
    max_tokens:
        Optional hard ceiling on noise tokens.  Defaults to the model's
        context window.
    on_step:
        Optional callback invoked after each step with the :class:`StepResult`.

    Returns
    -------
    EvaluationResult
    """
    ceiling = max_tokens or llm_client.max_context_window
    history: List[StepResult] = []
    adt: Optional[int] = None

    # ── 1. Rank files (done once) ────────────────────────────────────────
    analysis: AnalysisResult = polysemy_analyzer.analyze()

    # ── 2. Progressive loop ──────────────────────────────────────────────
    n = step_size
    while n <= ceiling:
        # 2a. Build noisy prompt.
        injected: InjectedPrompt = context_injector.build_prompt(
            ranked_files=analysis.ranked_files,
            task_prompt=task_config.task_description,
            constraint_string=task_config.constraint_prompt,
            target_noise_tokens=n,
        )

        # 2b. Generate code.
        generated_code: str = llm_client.generate(injected.prompt)

        # 2c. Strip markdown fences if the model wrapped its output.
        generated_code = _strip_code_fences(generated_code)

        # 2d. Verify with weaver.
        weaver_result: WeaverResult = calculate_sas(
            code_string=generated_code,
            forbidden_imports=task_config.forbidden_imports,
            forbidden_globals=task_config.forbidden_globals,
        )

        step = StepResult(
            noise_tokens=n,
            sas=weaver_result.sas,
            violations=weaver_result.violations,
            files_included=injected.files_included,
            total_prompt_tokens=injected.total_token_count,
        )
        history.append(step)

        if on_step:
            on_step(step)

        # 2e. Check result.
        if weaver_result.sas == 0:
            adt = n
            break

        n += step_size

    return EvaluationResult(
        model_name=model_name,
        api_score=analysis.api_score,
        adt_threshold=adt,
        history=history,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```python … ```)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove the opening fence line.
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()
    return stripped
