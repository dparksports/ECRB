"""
core/context_injector.py — The Noise Phase
───────────────────────────────────────────
Assembles the final LLM prompt by concatenating "Homogeneous Noise" files
(ranked by polysemy) until an exact token budget is reached, then appending
the task prompt and structural constraint.

Token counting uses tiktoken (cl100k_base).

Architectural constraint  [⍜ STRICT_ISOLATION]:
  • No global state.  All parameters injected via function arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import tiktoken


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InjectedPrompt:
    """The fully-assembled prompt returned by the injector.

    Attributes
    ----------
    prompt:
        The concatenation of ``[NOISE] + [TASK] + [CONSTRAINT]``.
    noise_token_count:
        Exact number of noise tokens included.
    total_token_count:
        Total tokens across the entire prompt.
    files_included:
        Number of noise files that fit within the budget.
    """

    prompt: str
    noise_token_count: int
    total_token_count: int
    files_included: int


# ── Context Injector ─────────────────────────────────────────────────────────

class ContextInjector:
    """Builds prompts with a precise token budget of background noise.

    Parameters
    ----------
    encoding_name:
        The tiktoken encoding to use.  Defaults to ``cl100k_base``.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoder = tiktoken.get_encoding(encoding_name)

    # ── Public API ───────────────────────────────────────────────────────

    def build_prompt(
        self,
        ranked_files: List[Tuple[str, float]],
        task_prompt: str,
        constraint_string: str,
        target_noise_tokens: int,
    ) -> InjectedPrompt:
        """Construct the noise-injected prompt.

        Parameters
        ----------
        ranked_files:
            Ordered list of ``(file_path, similarity_score)`` tuples.
        task_prompt:
            The natural-language task the model must complete.
        constraint_string:
            The structural constraint block appended at the end.
        target_noise_tokens:
            Maximum number of noise tokens to include.

        Returns
        -------
        InjectedPrompt
        """
        noise_block, noise_tokens, files_used = self._assemble_noise(
            ranked_files, target_noise_tokens
        )

        # ── Compose the final prompt ─────────────────────────────────────
        sections = [
            "=" * 72,
            "BACKGROUND CONTEXT (Repository Files)",
            "=" * 72,
            noise_block,
            "",
            "=" * 72,
            "TASK",
            "=" * 72,
            task_prompt,
            "",
            "=" * 72,
            "STRUCTURAL CONSTRAINT",
            "=" * 72,
            constraint_string,
        ]

        full_prompt = "\n".join(sections)
        total_tokens = len(self._encoder.encode(full_prompt))

        return InjectedPrompt(
            prompt=full_prompt,
            noise_token_count=noise_tokens,
            total_token_count=total_tokens,
            files_included=files_used,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _assemble_noise(
        self,
        ranked_files: List[Tuple[str, float]],
        budget: int,
    ) -> Tuple[str, int, int]:
        """Read and concatenate files until the token budget is consumed.

        Returns ``(noise_text, actual_token_count, files_used)``.
        """
        chunks: List[str] = []
        consumed: int = 0
        files_used: int = 0

        for file_path, score in ranked_files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                continue

            header = f"\n# ── FILE: {file_path} (similarity={score}) ──\n"
            segment = header + content
            segment_tokens = self._count_tokens(segment)

            if consumed + segment_tokens > budget:
                # Truncate the last file to fit exactly.
                remaining = budget - consumed
                if remaining > 0:
                    truncated = self._truncate_to_tokens(segment, remaining)
                    chunks.append(truncated)
                    consumed += self._count_tokens(truncated)
                    files_used += 1
                break

            chunks.append(segment)
            consumed += segment_tokens
            files_used += 1

        return "\n".join(chunks), consumed, files_used

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate *text* to at most *max_tokens*."""
        token_ids = self._encoder.encode(text)[:max_tokens]
        return self._encoder.decode(token_ids)
