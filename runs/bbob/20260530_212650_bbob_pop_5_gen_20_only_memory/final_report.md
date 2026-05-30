# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000049` | valid | 0.526593 | adaptive_hybrid_cma_pattern |
| 2 | `cand_000045` | valid | 0.524054 | intensification_cma_with_pattern |
| 3 | `cand_000027` | valid | 0.514138 | m6_diversify_search |
| 4 | `cand_000038` | valid | 0.507802 | explorative_cma_restart |
| 5 | `cand_000017` | valid | 0.503574 | intensify_hybrid |

## Search Best Candidate

- ID: `cand_000049`
- Name: adaptive_hybrid_cma_pattern
- Status: valid
- Search Mean AOCC: 0.5265926732726235
- Thought: This optimizer uses an exploitation-focused CMA-ES with doubled learning rates and a dynamic switch to a pattern search with random leaps when stagnation is detected. The CMA-ES phase uses large population (10+3*log(dim)), reduced initial step size (0.2*range), and doubled covariance learning rates for fast adaptation. If no improvement for 10% of budget, it switches to the local search phase, which performs coordinate-wise pattern search with a decreasing step size, and occasionally adds random leaps to escape local minima. The random seed controls all stochastic decisions.

## LLM Calls

- Candidate-generation calls: 51
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 51
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 125
- Budget match: False
- LLM reflections enabled: False
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 0
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 32 / 64
- History buckets: 39
- Added candidates: 40
- Duplicate rejections: 0
- History parent selections: 22
- History offspring with history parent: 20
- Final selection from history: False

## Committee

- Output mode: committee_specialist
- Committee size: 3

### Specialists

**cand_000025** (diverse_restart_cma)
- Status: valid
- Score: 0.5143722865320702
- Assigned instances: 8 — 15, 16, 17, 19, 21, 22, 4, 5
- Code: import numpy as np
import math

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
...

**cand_000049** (adaptive_hybrid_cma_pattern)
- Status: valid
- Score: 0.5164066881776618
- Assigned instances: 13 — 1, 10, 11, 12, 13, 14, 2, 20, 23, 24 ...
- Code: import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call_...

**cand_000027** (m6_diversify_search)
- Status: valid
- Score: 0.5348620279376554
- Assigned instances: 3 — 18, 7, 9
- Code: class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed...

### VBS (Virtual Best Solver)

- VBS mean score: 0.610943
- VBS per-instance scores: {'1': 0.9573715681208158, '10': 0.8307221106882027, '11': 0.8517052159662324, '12': 0.7820423844597255, '13': 0.8208284081984997, '14': 0.912508100547499, '15': 0.15254322871587017, '16': 0.8575611497016603, '17': 0.8556006550657382, '18': 0.3485377514962653, '19': 0.22136860815876983, '2': 0.8479365123218718, '20': 0.1994421913617313, '21': 0.5981629864801558, '22': 0.4038925451441806, '23': 0.23336985954901907, '24': 0.1048094447487953, '3': 0.15855477010679503, '4': 0.12160472506863074, '5': 0.9919913882674258, '6': 0.8895825062530522, '7': 0.946418637129012, '8': 0.8202226290515988, '9': 0.7558562312581796}

