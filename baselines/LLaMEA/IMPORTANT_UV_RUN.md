# ⚠️ IMPORTANT: Use `uv run` to Execute Scripts

## The Correct Way to Run

When using the LLaMEA framework with `uv` package manager, **always use `uv run` to execute Python scripts**.

### ✅ CORRECT:
```bash
cd baselines/LLaMEA
uv sync --dev
export DEEPSEEK_API_KEY="sk-..."
uv run python examples/tsp-gls-hpo-deepseek.py
```

### ❌ WRONG (will cause numpy/dependency errors):
```bash
cd baselines/LLaMEA
python examples/tsp-gls-hpo-deepseek.py  # ❌ Dependencies not loaded!
```

## Why This Matters

- **`uv sync --dev`** creates a virtual environment with all dependencies
- **`uv run python`** executes scripts within that virtual environment
- **Just `python`** runs the system Python without the virtual environment's dependencies

## Without `uv run`, you'll get errors like:
```
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'ConfigSpace'
ModuleNotFoundError: No module named 'smac'
```

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `uv sync --dev` |
| Run OpenAI version | `uv run python examples/tsp-gls-hpo.py` |
| Run DeepSeek version | `uv run python examples/tsp-gls-hpo-deepseek.py` |
| Run other Python scripts | `uv run python <script.py>` |

## Summary

Always remember:
- After `uv sync --dev`, use `uv run python` for all script execution
- This ensures all dependencies from `pyproject.toml` are available
- Direct `python` command skips the virtual environment
