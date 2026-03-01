"""
core/polysemy_analyzer.py — Adversarial Polysemy Engine
────────────────────────────────────────────────────────
Calculates the Adversarial Polysemy Index (API) for a codebase relative to
a target task string.  Files are ranked by TF-IDF cosine similarity so the
most lexically overlapping files appear first — these act as "Homogeneous
Noise" that floods the LLM's context window.

Architectural constraint  [⍜ STRICT_ISOLATION]:
  • No global state.  The analyzer is instantiated with explicit parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity     # type: ignore[import-untyped]


# ── Constants ────────────────────────────────────────────────────────────────

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".c", ".cpp", ".h"}
)


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalysisResult:
    """Immutable result of the polysemy analysis.

    Attributes
    ----------
    ranked_files:
        Ordered list of ``(file_path, similarity_score)`` tuples,
        sorted descending by score.
    api_score:
        The Adversarial Polysemy Index — mean cosine similarity across
        all analysed files.  Higher ⇒ more homogeneous noise in the repo.
    """

    ranked_files: List[Tuple[str, float]]
    api_score: float


# ── Analyzer ─────────────────────────────────────────────────────────────────

class PolysemyAnalyzer:
    """Ranks repository files by lexical overlap with a task description.

    Parameters
    ----------
    repo_path:
        Path to the root of the target codebase.
    task_string:
        The natural-language task description to compare against.
    extensions:
        Optional set of file extensions to include.  Defaults to common
        source-code extensions.
    """

    def __init__(
        self,
        repo_path: str | Path,
        task_string: str,
        extensions: frozenset[str] | None = None,
    ) -> None:
        self._repo_path = Path(repo_path)
        self._task_string = task_string
        self._extensions = extensions or _SUPPORTED_EXTENSIONS

    # ── Public API ───────────────────────────────────────────────────────

    def analyze(self) -> AnalysisResult:
        """Run the TF-IDF analysis and return an :class:`AnalysisResult`.

        Returns
        -------
        AnalysisResult
            Ranked file list and the aggregate API score.

        Raises
        ------
        FileNotFoundError
            If *repo_path* does not exist.
        ValueError
            If no eligible source files are found in the repository.
        """
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: {self._repo_path}"
            )

        file_paths, file_contents = self._collect_files()

        if not file_paths:
            raise ValueError(
                f"No source files with extensions "
                f"{sorted(self._extensions)} found in {self._repo_path}"
            )

        ranked_files, api_score = self._rank_by_similarity(
            file_paths, file_contents
        )

        return AnalysisResult(ranked_files=ranked_files, api_score=api_score)

    # ── Private helpers ──────────────────────────────────────────────────

    def _collect_files(self) -> Tuple[List[str], List[str]]:
        """Walk the repo and return parallel lists of paths and contents."""
        paths: List[str] = []
        contents: List[str] = []

        for root, _dirs, files in os.walk(self._repo_path):
            for fname in files:
                if Path(fname).suffix in self._extensions:
                    full_path = os.path.join(root, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                            text = fh.read()
                    except (OSError, UnicodeDecodeError):
                        continue

                    if text.strip():
                        paths.append(full_path)
                        contents.append(text)

        return paths, contents

    def _rank_by_similarity(
        self,
        file_paths: List[str],
        file_contents: List[str],
    ) -> Tuple[List[Tuple[str, float]], float]:
        """Compute TF-IDF cosine similarity and return ranked list + API score."""

        # The corpus is [task_string, file_0, file_1, …]
        corpus = [self._task_string] + file_contents

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10_000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Similarity of each file against the task string (row 0).
        similarities = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]
        ).flatten()

        # Pair each path with its similarity score.
        scored: List[Tuple[str, float]] = list(
            zip(file_paths, [round(float(s), 6) for s in similarities])
        )

        # Sort descending by score.
        scored.sort(key=lambda x: x[1], reverse=True)

        api_score = round(float(similarities.mean()), 6) if len(similarities) else 0.0

        return scored, api_score
