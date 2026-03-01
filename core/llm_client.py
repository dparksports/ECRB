"""
core/llm_client.py — LLM Client Factory
─────────────────────────────────────────
Reads model_config.yaml and provides a standardised interface to route
prompts to any configured LLM provider via litellm.

Architectural constraint  [⍜ STRICT_ISOLATION]:
  • Zero global state — configuration is injected through the constructor.
  • The factory returns a self-contained LLMClient instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# litellm is an optional runtime dependency; we import lazily so the module
# can still be loaded (and tested structurally) without it installed.
try:
    import litellm  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    litellm = None  # type: ignore[assignment]


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """Immutable snapshot of a single model's configuration."""

    provider: str
    model_id: str
    api_key_env_var: str
    endpoint_url: str
    max_context_window: int


@dataclass(frozen=True)
class GlobalDefaults:
    """Immutable default generation parameters."""

    temperature: float = 0.0
    max_output_tokens: int = 4096
    token_step_size: int = 10_000


# ── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """A configured, ready-to-use LLM client for a specific model.

    All state is injected via the constructor — no global mutation occurs.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        defaults: GlobalDefaults,
    ) -> None:
        self._config = model_config
        self._defaults = defaults

        # Resolve the API key from the environment at construction time.
        self._api_key: Optional[str] = os.environ.get(
            self._config.api_key_env_var
        )

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._config.model_id

    @property
    def max_context_window(self) -> int:
        return self._config.max_context_window

    def generate(self, prompt_payload: str) -> str:
        """Send *prompt_payload* to the model and return the generated code.

        Parameters
        ----------
        prompt_payload:
            The fully-assembled prompt string (noise + task + constraint).

        Returns
        -------
        str
            The raw generated code string from the model.

        Raises
        ------
        RuntimeError
            If litellm is not installed or the API key is missing.
        """
        if litellm is None:
            raise RuntimeError(
                "litellm is not installed. "
                "Run `pip install litellm` to enable LLM calls."
            )

        if not self._api_key:
            raise RuntimeError(
                f"API key not found. Set the environment variable "
                f"'{self._config.api_key_env_var}' before running."
            )

        # Set the key for litellm to pick up.
        os.environ[self._config.api_key_env_var] = self._api_key

        response = litellm.completion(
            model=self._config.model_id,
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=self._defaults.temperature,
            max_tokens=self._defaults.max_output_tokens,
            api_base=self._config.endpoint_url,
        )

        return response.choices[0].message.content  # type: ignore[union-attr]


# ── Factory ──────────────────────────────────────────────────────────────────

class LLMClientFactory:
    """Constructs :class:`LLMClient` instances from a YAML config file.

    Parameters
    ----------
    config_path:
        Absolute or relative path to ``model_config.yaml``.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._raw: Dict[str, Any] = self._load_config()
        self._defaults = self._parse_defaults()

    # ── Public API ───────────────────────────────────────────────────────

    def get_client(self, model_name: str) -> LLMClient:
        """Return an :class:`LLMClient` for the requested *model_name*.

        Raises
        ------
        KeyError
            If *model_name* is not found in the configuration file.
        """
        models: Dict[str, Any] = self._raw.get("models", {})
        if model_name not in models:
            available = ", ".join(sorted(models.keys()))
            raise KeyError(
                f"Model '{model_name}' not found in config. "
                f"Available models: {available}"
            )

        entry = models[model_name]
        model_config = ModelConfig(
            provider=entry["provider"],
            model_id=entry["model_id"],
            api_key_env_var=entry["api_key_env_var"],
            endpoint_url=entry["endpoint_url"],
            max_context_window=entry["max_context_window"],
        )

        return LLMClient(model_config=model_config, defaults=self._defaults)

    def list_models(self) -> list[str]:
        """Return all model names defined in the config."""
        return sorted(self._raw.get("models", {}).keys())

    # ── Private helpers ──────────────────────────────────────────────────

    def _load_config(self) -> Dict[str, Any]:
        with open(self._config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _parse_defaults(self) -> GlobalDefaults:
        raw_defaults = self._raw.get("defaults", {})
        return GlobalDefaults(
            temperature=raw_defaults.get("temperature", 0.0),
            max_output_tokens=raw_defaults.get("max_output_tokens", 4096),
            token_step_size=raw_defaults.get("token_step_size", 10_000),
        )
