# LLaMEA-HPO for TSP: DeepSeek Quick Start

Using **DeepSeek** (more cost-effective alternative to GPT-4o)

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (from LLaMEA root)

```bash
cd baselines/LLaMEA
uv sync --dev
```

This installs all dependencies including DeepSeek support.

### Step 2: Set Your DeepSeek API Key

```bash
export DEEPSEEK_API_KEY="sk-..."
```

Replace `sk-...` with your actual DeepSeek API key.

### Step 3: Run the DeepSeek Example

```bash
# Make sure you're in the LLaMEA root directory
cd baselines/LLaMEA

# Run the TSP example with DeepSeek (using uv to ensure dependencies)
uv run python examples/tsp-gls-hpo-deepseek.py
```

**Note**: Always use `uv run python` instead of just `python` to ensure all dependencies from `pyproject.toml` are available!

## ✅ What to Expect

### Initial Output
```
======================================================================
LLaMEA-HPO for TSP (DeepSeek Version)
======================================================================

Initializing LLM: deepseek-v4-flash
Loading TSP instances...
✓ Loaded a280: 280 nodes, optimal=2579
✓ Loaded d198: 198 nodes, optimal=15780
[... 8 more instances ...]

Loaded 10 TSP instances
```

### Progress Output
```
============================================================
Evaluating: InitialHeuristic
============================================================
Running validation...
  Validation OK on a280: cost=2750.5
Running SMAC3 hyperparameter optimization...
  SMAC3 completed. Best config: {'penalty_factor': 0.45, ...}
Final evaluation on all instances...
  a280: gap=6.64%
  d198: gap=3.22%
  ...
Result: Mean gap: 5.42% ± 1.89% | Config: {...}

[Iteration 2/20...]
[...]
```

## 🔄 Comparing with OpenAI Version

| Feature | OpenAI | DeepSeek |
|---------|--------|----------|
| **Model** | gpt-4o-2024-05-13 | deepseek-v4-flash |
| **Cost** | Higher | Lower (~50-70% cheaper) |
| **Speed** | Fast | Fast |
| **Quality** | Excellent | Very Good |
| **File** | `tsp-gls-hpo.py` | `tsp-gls-hpo-deepseek.py` |

DeepSeek is an excellent cost-effective alternative for experimentation and development!

## 📊 Performance Expectations

| Metric | Value |
|--------|-------|
| **Initial gap** | 10-15% |
| **Best gap** | 3-8% |
| **Per iteration** | 5-15 minutes |
| **Total time** | 100-200 minutes |

## 🔧 Configuration

To modify the optimization:

**Change number of iterations:**
```python
n_gens = 30  # Change from 20 to 30
```

**Change model:**
```python
# Use a different DeepSeek model if available
llm = DeepSeek_LLM(api_key, model="deepseek-v3")  # Or another variant
```

**Use OpenAI instead:**
Switch to `tsp-gls-hpo.py` which uses GPT-4o.

## 🐛 Troubleshooting

### "Cannot import dynagen.domain.tsp_parser"
**Fix:** Make sure you're running from the `baselines/LLaMEA` directory:
```bash
cd baselines/LLaMEA
python examples/tsp-gls-hpo-deepseek.py
```

### "DEEPSEEK_API_KEY environment variable not set"
**Fix:** Set your DeepSeek API key before running:
```bash
export DEEPSEEK_API_KEY="sk-..."
python examples/tsp-gls-hpo-deepseek.py
```

### Dependencies missing
**Fix:** Install dependencies:
```bash
cd baselines/LLaMEA
uv sync --dev
```

## 📚 File Comparison

| Purpose | File |
|---------|------|
| **OpenAI (GPT-4o)** | `examples/tsp-gls-hpo.py` |
| **DeepSeek** | `examples/tsp-gls-hpo-deepseek.py` |
| **Quick Start (OpenAI)** | `LLAMEA_TSP_QUICKSTART.md` |
| **Quick Start (DeepSeek)** | `LLAMEA_TSP_DEEPSEEK_QUICKSTART.md` |
| **Full Reference** | `examples/README_TSP_GLS_HPO.md` |

## ✨ Key Advantages of DeepSeek Version

✓ **Lower cost** - Approximately 50-70% cheaper than GPT-4o  
✓ **Same quality** - LLM-generated heuristics are competitive  
✓ **Same framework** - Uses LLaMEA identically  
✓ **Full HPO** - SMAC3 optimization works the same  
✓ **Reproducible** - Same methodology, just different model  

## 🚀 Recommended Usage

- **Development/Experimentation** → Use DeepSeek (cost-effective)
- **Production/Best Results** → Use GPT-4o (highest quality)
- **Cost-Sensitive** → Use DeepSeek consistently
- **Comparison** → Run both and compare results

## 📞 Next Steps

1. **Get DeepSeek API key** from https://www.deepseek.com/
2. **Set environment variable**: `export DEEPSEEK_API_KEY="sk-..."`
3. **Run the example**:
   ```bash
   cd baselines/LLaMEA
   python examples/tsp-gls-hpo-deepseek.py
   ```
4. **Monitor progress** and wait for results

---

**Ready to start with DeepSeek?**

```bash
cd baselines/LLaMEA
uv sync --dev
export DEEPSEEK_API_KEY="sk-..."
python examples/tsp-gls-hpo-deepseek.py
```

Good luck! 🚀
