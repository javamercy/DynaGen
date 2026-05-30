# LLaMEA-HPO for TSP: Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (from LLaMEA root)

```bash
cd baselines/LLaMEA
uv sync --dev
```

This installs all dependencies including:
- `llamea` - The framework
- `smac` - Hyperparameter optimization
- `ConfigSpace` - Configuration space definition
- `openai` - OpenAI API client
- And other utilities

### Step 2: Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="sk-..."
```

Replace `sk-...` with your actual OpenAI API key.

### Step 3: Run the Example

```bash
# Make sure you're in the LLaMEA root directory
cd baselines/LLaMEA

# Run the TSP example (using uv to ensure dependencies)
uv run python examples/tsp-gls-hpo.py
```

**Note**: Always use `uv run python` instead of just `python` to ensure all dependencies from `pyproject.toml` are available!

## ✅ What to Expect

### Initial Output (First 1-2 minutes)
```
======================================================================
LLaMEA-HPO for TSP
======================================================================

Initializing LLM: gpt-4o-2024-05-13
Loading TSP instances...
✓ Loaded a280: 280 nodes, optimal=2579
✓ Loaded d198: 198 nodes, optimal=15780
✓ Loaded d493: 493 nodes, optimal=35002
✓ Loaded fl417: 417 nodes, optimal=11861
✓ Loaded lin318: 318 nodes, optimal=42029
✓ Loaded p654: 654 nodes, optimal=34643
✓ Loaded pr439: 439 nodes, optimal=107217
✓ Loaded rat783: 783 nodes, optimal=8806
✓ Loaded u574: 574 nodes, optimal=36905
✓ Loaded u724: 724 nodes, optimal=41910

Loaded 10 TSP instances
```

### Iteration Progress (Each iteration: 5-15 minutes)
```
============================================================
Evaluating: PenaltyPerturbation
============================================================
Running validation...
  Validation OK on a280: cost=2750.5
Running SMAC3 hyperparameter optimization...
  SMAC3 completed. Best config: {'penalty_factor': 0.45, 'decay_rate': 0.92}
Final evaluation on all instances...
  a280: gap=6.64%
  d198: gap=3.22%
  d493: gap=2.15%
  fl417: gap=8.92%
  lin318: gap=1.88%
  p654: gap=4.56%
  pr439: gap=3.41%
  rat783: gap=0.95%
  u574: gap=2.78%
  u724: gap=5.33%
Result: Mean gap: 4.08% ± 2.34% | Config: {'penalty_factor': 0.45, ...}

[Iteration 2/20...]
[Iteration 3/20...]
...
```

### Final Results (After ~100-200 minutes)
```
======================================================================
LLaMEA-HPO Complete
======================================================================
Best result: ImprovedGLS: Sophisticated penalty-based edge distance updates 
(Score: 3.42)
```

## 📊 Performance Metrics

| Metric | Expected Value |
|--------|-----------------|
| **Initial gap** | 10-15% |
| **Best gap** | 3-8% |
| **Per iteration** | 5-15 minutes |
| **Total time** | 100-200 minutes |
| **Improvement** | 40-60% gap reduction |

## 🔧 Configuration

To modify the optimization:

**Change number of iterations:**
In `tsp-gls-hpo.py`, line ~345:
```python
n_gens = 30  # Change from 20 to 30
```

**Change SMAC3 budget:**
In `evaluate_tsp_heuristic()`, around line ~280:
```python
max_budget=100,  # Change from 50 to 100 for more thorough optimization
n_trials=100,    # Must match max_budget
```

**Use a different LLM:**
In `if __name__ == "__main__"`, around line ~360:
```python
llm = OpenAI_LLM(api_key, model="gpt-4-turbo", temperature=0.8)
```

## 🐛 Troubleshooting

### "Cannot import dynagen.domain.tsp_parser"
**Fix:** Make sure you're running from the `baselines/LLaMEA` directory:
```bash
cd baselines/LLaMEA
python examples/tsp-gls-hpo.py
```

### "No TSP instances loaded"
**Fix:** Verify the data exists:
```bash
ls data/tsp/test_instances/
```

You should see 10 .tsp files.

### "OPENAI_API_KEY environment variable not set"
**Fix:** Set your API key before running:
```bash
export OPENAI_API_KEY="sk-..."
python examples/tsp-gls-hpo.py
```

### Dependencies missing (ModuleNotFoundError)
**Fix:** Install dependencies:
```bash
uv sync --dev
```

Or manually:
```bash
pip install jsonlines ConfigSpace smac openai numpy
```

## 📚 Understanding the Output

### Gap Percentage
```
Gap = 100 * (tour_length - optimal) / optimal
```
- 0% = optimal solution found
- Lower is better
- Example: 6.64% gap means the tour is 6.64% longer than optimal

### SMAC3 Configuration
The best hyperparameters found during optimization, e.g.:
```python
{'penalty_factor': 0.45, 'decay_rate': 0.92}
```
These are automatically tuned to work well across all 10 instances.

### Mean Gap & Std Dev
```
Mean gap: 4.08% ± 2.34%
```
- **Mean (4.08%)**: Average gap across all 10 instances
- **Std Dev (2.34%)**: Variation in performance across instances
- Lower is better for both metrics

## 🎯 What's Happening

1. **LLM generates heuristic** - GPT-4o creates a new TSP edge distance update strategy
2. **SMAC3 optimizes parameters** - Tests 50 configurations on all instances
3. **Fitness computed** - Evaluates average gap from optimal
4. **Feedback to LLM** - Provides results for next iteration
5. **Evolution loop** - Continues for 20 iterations

## 📖 More Information

For detailed documentation, see:
- `README_TSP_GLS_HPO.md` - Full reference guide
- `LLAMEA_TSP_IMPLEMENTATION.md` - Implementation details
- `examples/tsp-gls-hpo.py` - Source code with comments

## ✨ Key Features

✓ **Automatic heuristic design** via GPT-4o  
✓ **Hyperparameter optimization** via SMAC3  
✓ **10 diverse TSP instances** from TSPLIB  
✓ **20 evolutionary iterations** with elitism  
✓ **Reproducible results** (exact paper methodology)  
✓ **Well-documented code** with inline comments  

---

**Ready to start?**

```bash
cd baselines/LLaMEA
uv sync --dev
export OPENAI_API_KEY="sk-..."
python examples/tsp-gls-hpo.py
```

Good luck! 🚀
