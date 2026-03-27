# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Heretic?

Heretic is a fully automatic censorship removal tool for transformer-based language models. It uses directional ablation (abliteration) combined with Optuna TPE-based hyperparameter optimization to create decensored models. The core math: ΔW = -λ · v · (vᵀW), applied via rank-1 LoRA adapters for efficiency.

## Build & Development Commands

Package manager is **uv**. Entry point: `heretic = heretic.main:main`.

```bash
# Install in development mode
uv sync

# Install with research/visualization extras
uv sync --extra research

# Format
ruff format .

# Lint (with import sorting)
ruff check --extend-select I

# Type check (strict)
ty check --error-on-warning

# Build package
uv build
```

There is no test suite — quality is enforced via formatting, linting, and type checking. CI runs these across Python 3.10–3.13.

## Architecture

Six modules in `src/heretic/`, ~3000 lines total:

- **`main.py`** — CLI entry point and optimization orchestrator. Handles hardware detection, batch size tuning, refusal direction computation, Optuna trial loop (bi-objective: minimize refusals + KL divergence), and a post-processing menu (save, upload, chat, benchmark, compare).

- **`model.py`** — `Model` class wrapping transformers. Handles dtype fallback chains, 4-bit quantization via bitsandbytes, LoRA-based abliteration (rank-1 adapters), batched residual extraction, inference, and logprob computation. Supports floating refusal direction indices with linear interpolation and flexible ablation weight kernels.

- **`config.py`** — Pydantic Settings with layered config: CLI args > env vars (`HERETIC_` prefix) > TOML files. `Settings` is the master config class (~30 parameters). `config.default.toml` has production defaults.

- **`evaluator.py`** — Measures abliteration quality via refusal counting (pattern-matched against ~31 markers) and KL divergence from the original model on harmless prompts.

- **`analyzer.py`** — Research features: residual geometry statistics (cosine similarity, L2 norms, silhouette coefficients) and PaCMAP visualizations. Requires `research` extras.

- **`utils.py`** — Helpers: `Prompt` dataclass, HuggingFace dataset loading, interactive UI (questionary), memory reporting, batching utilities.

## Key Design Patterns

- Abliteration is applied as LoRA adapters (never modifies base weights directly), enabling fast resets by zeroing adapter weights.
- Model supports quantized inference with dequantization for ablation computation.
- Optimization state persists via Optuna JournalStorage (JSON), enabling interrupted run resumption.
- Hardware detection supports CUDA, XPU, NPU, MLU, MPS, and CPU fallback.
