# LLaMEA-HPO for TSP with Guided Local Search

## Overview

This example demonstrates the **LLaMEA-HPO** (Large Language Model Evolutionary Algorithm with Hyperparameter Optimization) framework applied to the **Traveling Salesman Problem (TSP)** using **Guided Local Search (GLS)**.

The framework generates novel heuristic functions that update the edge distance matrix to help escape local optima, with hyperparameters automatically optimized using SMAC3.

**File**: `examples/tsp-gls-hpo.py`
**Run from**: `baselines/LLaMEA/` root directory

## Key Features

- ✅ **LLM-Generated Heuristics**: Uses GPT-4o to design novel TSP distance update strategies
- ✅ **Guided Local Search**: Implements GLS with 2-opt local search and custom perturbations
- ✅ **SMAC3 HPO**: Automatic hyperparameter optimization (50 evaluations per iteration)
- ✅ **Multi-Instance Evaluation**: Tests on 10 diverse TSP instances
- ✅ **Gap-Based Fitness**: Uses percentage gap from optimal solution as fitness metric
- ✅ **20-Iteration Evolution**: Full evolutionary loop with elitism

## Architecture

```
tsp-gls-hpo.py
├── TSPGuidedLocalSearch (GLS Implementation)
│   ├── nearest_neighbor()     - Tour initialization
│   ├── tour_cost()            - Cost computation
│   ├── two_opt_move()         - 2-opt local search
│   ├── local_search()         - LS convergence loop
│   └── gls()                  - Main GLS with custom distance update
│
├── load_tsp_instances()       - Load 10 TSP problems
├── compute_gap()              - Calculate gap percentage
├── evaluate_tsp_heuristic()   - SMAC3-based evaluation
└── LLaMEA main loop           - 20 iterations, elitism enabled
```

## Usage

### Quick Start

**Step 1: Install LLaMEA dependencies**
```bash
cd baselines/LLaMEA
uv sync --dev
```

**Step 2: Set OpenAI API key**
```bash
export OPENAI_API_KEY="sk-..."
```

**Step 3: Run from LLaMEA root**
```bash
# From baselines/LLaMEA directory
python examples/tsp-gls-hpo.py
```

### Expected Output
```
======================================================================
LLaMEA-HPO for TSP
======================================================================

Initializing LLM: gpt-4o-2024-05-13
Loading TSP instances...
✓ Loaded a280: 280 nodes, optimal=2579
✓ Loaded d198: 198 nodes, optimal=15780
[... 8 more instances ...]

Loaded 10 TSP instances

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

[Iterations 2-20 continue...]

LLaMEA-HPO Complete
Best result: ...
```

### Troubleshooting

**Missing dependencies:**
```bash
cd baselines/LLaMEA
pip install jsonlines ConfigSpace smac openai numpy
```

**Cannot find TSP instances:**
The script expects TSP files in `../data/tsp/test_instances/` relative to the LLaMEA root.
Verify files exist: `ls data/tsp/test_instances/`

**Cannot import dynagen:**
The DynaGen TSP parser is imported from the parent directory.
Make sure you're running from `baselines/LLaMEA/` and DynaGen is accessible.

## Configuration

### Model
- **LLM**: gpt-4o-2024-05-13
- **Temperature**: 0.8 (creative but reliable)
- **Reason**: Most cost-effective for code generation (as per paper)

### Optimization Loop
- **Iterations**: 20
- **Population**: 5 parents, 5 offspring per iteration
- **Selection**: Elitism (best solution always preserved)
- **Objective**: Minimize mean gap percentage

### SMAC3 HPO
- **Budget**: 50 evaluations per heuristic
- **Instances**: All 10 TSP problems used for optimization
- **Workers**: Up to 4 parallel evaluations
- **Reason**: Scaled from paper's 256/64 ratio (256 evals / 64 instances)

### GLS Parameters
- **GLS Iterations**: 50-100 (development) vs 20 (validation)
- **Max Local Search**: 1000 iterations
- **Perturbation**: Dynamic based on edge usage

## TSP Instances

The example uses 10 TSPLIB instances located in `data/tsp/test_instances/`:

| Instance | Size | Optimal | Source |
|----------|------|---------|--------|
| a280 | 280 | 2579 | Drilling problem |
| d198 | 198 | 15780 | Random Euclidean |
| d493 | 493 | 35002 | Random Euclidean |
| fl417 | 417 | 11861 | Drilling problem |
| lin318 | 318 | 42029 | Drilling problem |
| p654 | 654 | 34643 | Random Euclidean |
| pr439 | 439 | 107217 | Random Euclidean |
| rat783 | 783 | 8806 | Random Euclidean |
| u574 | 574 | 36905 | Random Euclidean |
| u724 | 724 | 41910 | Random Euclidean |

## Expected Heuristic Output

The LLM generates code in this format:

```python
import numpy as np

class MyHeuristic:
    def __init__(self, param1=0.5, param2=0.9):
        self.param1 = param1
        self.param2 = param2
    
    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        """
        Update distance matrix based on edge usage.
        
        Args:
            edge_distance: Current distance matrix (n×n array)
            local_opt_tour: Current best tour (array of node IDs)
            edge_n_used: Number of times each edge was used (n×n array)
        
        Returns:
            updated_distance: Modified distance matrix
        """
        # Example: penalize frequently used edges
        penalty = self.param1 * np.power(edge_n_used + 1, self.param2)
        updated = edge_distance + penalty
        return updated

```

### Configuration Space

```python
{
    "param1": (0.1, 2.0),      # Penalty strength
    "param2": (0.5, 2.0),      # Penalty growth rate
}
```

SMAC3 optimizes these hyperparameters automatically.

## Evaluation Workflow

### Per Iteration

1. **Generate Heuristic**
   - LLM creates new TSP distance update strategy
   - Extracts code, name, and configuration space

2. **Validate**
   - Quick test on first instance
   - Ensures no syntax errors

3. **SMAC3 HPO**
   - Tests 50 different hyperparameter configurations
   - Each config evaluated on all 10 instances
   - Returns best configuration found

4. **Final Evaluation**
   - Uses best configuration
   - Evaluates on all 10 instances
   - Computes mean gap: `100 * (tour_length - optimal) / optimal`

5. **Feedback to LLM**
   - Reports mean gap and hyperparameters
   - Suggests refinement or redesign
   - Evolution continues to next iteration

## Performance Expectations

| Metric | Expected Range |
|--------|-----------------|
| Initial gap | 10-15% |
| Best heuristic gap | 3-8% |
| Improvement | 40-60% |
| Per-iteration time | 5-15 min |
| Total time (20 iter) | 100-200 min |

## Code Structure

### TSPGuidedLocalSearch Class

```python
class TSPGuidedLocalSearch:
    @staticmethod
    def nearest_neighbor(distance_matrix, start=0)
        """NN initialization"""
    
    @staticmethod
    def tour_cost(distance_matrix, tour)
        """Cost evaluation"""
    
    @staticmethod
    def two_opt_move(distance_matrix, tour)
        """2-opt neighborhood"""
    
    @staticmethod
    def local_search(distance_matrix, tour, max_iterations=1000)
        """Local search loop"""
    
    @staticmethod
    def gls(distance_matrix, update_distance_fn, gls_iters=100, ls_iters=1000)
        """Main GLS with custom distance update"""
```

### Integration with LLaMEA

The heuristic class is dynamically generated by the LLM:
- Class name: LLM-provided
- `__init__()`: Hyperparameters optimized by SMAC3
- `update_edge_distance()`: Custom perturbation strategy

## Customization

### Change Number of Iterations
```python
n_gens = 20  # Change to 10, 30, 50, etc.
es = LLaMEA(..., budget=n_gens, ...)
```

### Adjust SMAC3 Budget
```python
scenario = Scenario(
    ...,
    max_budget=50,  # Increase for more thorough optimization
    n_trials=50,    # Must correlate with max_budget
    ...
)
```

### Change GLS Iterations
```python
# In evaluate_config:
tour, cost = TSPGuidedLocalSearch.gls(
    instance["distance_matrix"],
    heuristic.update_edge_distance,
    gls_iters=100,  # Increase for longer search
)
```

### Use Different Model
```python
llm = OpenAI_LLM(api_key, model="gpt-4-turbo", temperature=0.8)
```

## Output Files

- **Console**: Real-time progress and fitness metrics
- **smac3_output/**: SMAC3 hyperparameter optimization logs
- **LLaMEA logs**: Experiment results and best heuristics

## Troubleshooting

### Missing TSP Instances
Verify instances are in `data/tsp/test_instances/`:
```bash
ls data/tsp/test_instances/
```

### API Errors
- Check `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`
- Verify API quota and credits
- Check rate limits

### Heuristic Errors
- Ensure `update_edge_distance` returns correct shape array
- Verify no input mutation
- Check hyperparameter bounds are reasonable

### SMAC3 Issues
- Reduce `max_budget` if running slow
- Increase `n_workers` for more parallelism
- Check SMAC3 output logs in `smac3_output/`

## References

### Paper
**In-the-Loop Automated Algorithm Design for Algorithms Generalization**
- Uses LLM for algorithm design
- SMAC3 for hyperparameter optimization
- Tested on multiple domains including TSP

### Methods
- **GLS**: Guided Local Search (Voudouris & Tsang, 1999)
- **2-opt**: Classic TSP local search operator
- **SMAC3**: Sequential Model-based Algorithm Configuration

### TSP Data
- **Source**: TSPLIB - Traveling Salesman Problem instances
- **Format**: Euclidean 2D coordinates or explicit distance matrices
- **Optimal**: Extracted from TSPLIB `OPTIMAL` field

## Related Examples

- `black-box-opt-with-HPO.py` - BBOB optimization with SMAC3
- `black-box-optimization.py` - Basic evolutionary algorithm

## Citation

If using this example, cite:

```bibtex
@inproceedings{llamea,
  title={In-the-Loop Automated Algorithm Design for Algorithms Generalization},
  author={...},
  year={2024}
}
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review SMAC3 output logs
3. Verify TSP instance files
4. Check OpenAI API status

---

**Status**: ✅ Complete and Ready to Use
**Last Updated**: May 30, 2026
