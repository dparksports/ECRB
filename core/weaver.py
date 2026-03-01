"""
core/weaver.py — System 2 Verifier (Maxwell's Demon)
─────────────────────────────────────────────────────
Verifies generated Python code against structural constraints using the
built-in ``ast`` module.  No fuzzy matching, no AI — purely deterministic.

Architectural constraint  [⊸ DETERMINISTIC_VERIFICATION]:
  • Uses ONLY Python's ``ast`` module for code analysis.
  • Returns a binary Structural Adherence Score (SAS): 1 = pass, 0 = fail.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Set


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WeaverResult:
    """Immutable result of the structural verification.

    Attributes
    ----------
    sas:
        Structural Adherence Score.  ``1`` if the code is clean,
        ``0`` if any violation was detected.
    violations:
        Human-readable descriptions of each detected violation.
    """

    sas: int
    violations: List[str] = field(default_factory=list)


# ── AST Visitor ──────────────────────────────────────────────────────────────

class ConstraintVisitor(ast.NodeVisitor):
    """Walks the AST looking for forbidden imports and global references.

    Parameters
    ----------
    forbidden_imports:
        Module / object names that must not appear in ``import`` or
        ``from … import`` statements.
    forbidden_globals:
        Top-level variable names that must not be assigned or referenced
        at module scope.
    """

    def __init__(
        self,
        forbidden_imports: Set[str],
        forbidden_globals: Set[str],
    ) -> None:
        self._forbidden_imports = forbidden_imports
        self._forbidden_globals = forbidden_globals
        self.violations: List[str] = []

    # ── Import statements ────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name in self._forbidden_imports:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden import `import {alias.name}`"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        # Check the module being imported from.
        if node.module:
            top_module = node.module.split(".")[0]
            if top_module in self._forbidden_imports:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden import "
                    f"`from {node.module} import …`"
                )

        # Check each imported name.
        for alias in node.names:
            if alias.name in self._forbidden_imports:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden import of name "
                    f"`{alias.name}` from `{node.module}`"
                )
        self.generic_visit(node)

    # ── Global variable assignments ──────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # Only flag top-level (module-scope) assignments.
        if self._is_module_scope(node):
            for target in node.targets:
                self._check_target(target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        if self._is_module_scope(node):
            self._check_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if self._is_module_scope(node) and node.target:
            self._check_target(node.target, node.lineno)
        self.generic_visit(node)

    # ── Global name references ───────────────────────────────────────────

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in self._forbidden_globals:
            self.violations.append(
                f"Line {node.lineno}: Reference to forbidden global "
                f"`{node.id}`"
            )
        self.generic_visit(node)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_module_scope(node: ast.AST) -> bool:
        """Return True if *node* sits directly inside the module body.

        We detect this by checking whether the node has a ``_parent``
        attribute pointing to a :class:`ast.Module`.  The parent link is
        injected in :func:`calculate_sas` before the walk begins.
        """
        parent = getattr(node, "_parent", None)
        return isinstance(parent, ast.Module)

    def _check_target(self, target: ast.AST, lineno: int) -> None:
        if isinstance(target, ast.Name) and target.id in self._forbidden_globals:
            self.violations.append(
                f"Line {lineno}: Forbidden global variable "
                f"assignment `{target.id}`"
            )
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                self._check_target(elt, lineno)


# ── Public API ───────────────────────────────────────────────────────────────

def _inject_parents(tree: ast.AST) -> None:
    """Walk the AST and set ``_parent`` on every node."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


def calculate_sas(
    code_string: str,
    forbidden_imports: Optional[List[str]] = None,
    forbidden_globals: Optional[List[str]] = None,
) -> WeaverResult:
    """Parse *code_string* and check it against structural constraints.

    Parameters
    ----------
    code_string:
        The raw Python source returned by the LLM.
    forbidden_imports:
        Names that must not appear in import statements.
    forbidden_globals:
        Names that must not be assigned or referenced at module scope.

    Returns
    -------
    WeaverResult
        ``sas=1`` if the code is clean; ``sas=0`` with a list of violations
        otherwise.  If the code cannot be parsed at all, SAS is ``0``.
    """
    forbidden_imports = forbidden_imports or []
    forbidden_globals = forbidden_globals or []

    # ── 1. Parse ─────────────────────────────────────────────────────────
    try:
        tree = ast.parse(code_string)
    except SyntaxError as exc:
        return WeaverResult(
            sas=0,
            violations=[f"SyntaxError: {exc.msg} (line {exc.lineno})"],
        )

    # ── 2. Inject parent links for scope detection ───────────────────────
    _inject_parents(tree)

    # ── 3. Walk ──────────────────────────────────────────────────────────
    visitor = ConstraintVisitor(
        forbidden_imports=set(forbidden_imports),
        forbidden_globals=set(forbidden_globals),
    )
    visitor.visit(tree)

    # ── 4. Score ─────────────────────────────────────────────────────────
    if visitor.violations:
        return WeaverResult(sas=0, violations=visitor.violations)

    return WeaverResult(sas=1, violations=[])
