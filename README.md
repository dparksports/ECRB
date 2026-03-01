# ⍜ ECRB — Enterprise Codebase Regression Benchmark

> **Evaluate LLM resilience to Channel Capacity Saturation and Adversarial Polysemy in real-world software engineering tasks.**

ECRB is an open-source evaluation harness that progressively floods an LLM's context window with highly correlated repository files while demanding it generate code that adheres to strict structural constraints, verified by a deterministic AST parser.

---

## Architecture

```
ECRB/
├── ecrb_runner.py            # CLI entry point (click + rich)
├── core/
│   ├── llm_client.py         # LLM factory — routes to OpenAI / Anthropic / Gemini / vLLM
│   ├── polysemy_analyzer.py  # TF-IDF adversarial polysemy engine
│   ├── context_injector.py   # Token-counted noise injector (tiktoken)
│   ├── weaver.py             # System 2 AST verifier (Maxwell's Demon)
│   └── evaluator.py          # Progressive saturation loop
├── config/
│   └── model_config.yaml     # Multi-provider model definitions
├── custom_task/
│   └── migration_rules.json  # Sample task + weaver constraints
└── requirements.txt
```

### Key Concepts

| Term | Definition |
|------|-----------|
| **Adversarial Polysemy Index (API)** | Mean TF-IDF cosine similarity between repo files and the task — measures how "noisy" the codebase is relative to the task. |
| **Homogeneous Noise** | Repository files ranked by polysemy and injected into the prompt to saturate the context window. |
| **Structural Adherence Score (SAS)** | Binary score (0 or 1) from the AST verifier. `1` = code is clean; `0` = forbidden imports or globals detected. |
| **Attention Degradation Threshold (ADT)** | The noise token count at which the model first fails the SAS check — its breaking point. |

### Architectural Constraints

- **⍜ STRICT_ISOLATION** — Zero global state. All modules communicate via explicit dependency injection.
- **⊸ DETERMINISTIC_VERIFICATION** — The AST verifier (`weaver.py`) uses only Python's built-in `ast` module. No fuzzy matching, no AI.

---

## Installation

```bash
cd ECRB
pip install -r requirements.txt
```

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

---

## Usage

### Run the benchmark

```bash
python ecrb_runner.py evaluate \
  --model gpt-4o \
  --custom-repo /path/to/target/codebase \
  --target-task custom_task/migration_rules.json
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name from `model_config.yaml` | *required* |
| `--custom-repo` | Path to target repository | *required* |
| `--target-task` | Path to task JSON | *required* |
| `--config` | Path to `model_config.yaml` | `config/model_config.yaml` |
| `--step-size` | Token increment per step | `10000` |
| `--max-tokens` | Hard ceiling on noise tokens | model's context window |

### Example Output

```
┌══════════════════════════════════════════════════════════════┐
│  ⍜  ECRB — Progressive Saturation Benchmark                │
│  Model: gpt-4o   Repo: /path/to/codebase                   │
└══════════════════════════════════════════════════════════════┘

╭─────────────── Saturation Steps ────────────────╮
│ Step │ Noise Tokens │ Prompt Tokens │ SAS │      │
│  1   │     10,000   │     11,234    │ 1 ✓ │ PASS │
│  2   │     20,000   │     21,456    │ 1 ✓ │ PASS │
│  3   │     30,000   │     31,678    │ 0 ✗ │ FAIL │
╰─────────────────────────────────────────────────╯

┃              ⍜ ECRB Results                       ┃
┃ Model  │ API Score │ ADT               │ Status   ┃
┃ gpt-4o │ 0.3421    │ 30,000 tokens     │ COLLAPSED┃
```

---

## Custom Tasks

Create a JSON file with this structure:

```json
{
  "task_description": "Your task description here.",
  "constraint_prompt": "Constraint ⍜: Your structural constraint here.",
  "weaver_rules": {
    "forbidden_imports": ["SomeModule", "AnotherModule"],
    "forbidden_globals": ["SOME_GLOBAL", "_instance"]
  }
}
```

---

## Adding Models

Edit `config/model_config.yaml` to add new model endpoints:

```yaml
models:
  my-custom-model:
    provider: openai
    model_id: openai/my-model
    api_key_env_var: MY_API_KEY
    endpoint_url: http://localhost:8000/v1
    max_context_window: 32000
```

---

## License

MIT
