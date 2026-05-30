# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000017` | valid | 0.532199 | de_curr2best_intensify |
| 2 | `cand_000009` | valid | 0.520208 | de_best_1_bin |
| 3 | `cand_000026` | valid | 0.487209 | intensified_local_search_es |
| 4 | `cand_000018` | valid | 0.474519 | de_best1_enhanced_local |
| 5 | `cand_000014` | valid | 0.446698 | de_best1_intensify |

## Search Best Candidate

- ID: `cand_000017`
- Name: de_curr2best_intensify
- Status: valid
- Search Mean AOCC: 0.5321986384761271
- Thought: Exploitation-focused variant of DE/best/1/intensify. Uses DE/current-to-best/1/bin for stronger attraction to the best solution. Population size reduced to max(4, min(5*dim, budget//5)) to allocate more budget to local search. Mutation factor F decreases linearly from 0.8 to 0.2 over generations. After DE, performs a pattern search along each dimension with decaying step, then random perturbations with exponentially decaying step size. All points clipped to bounds. Seed controls random number generator. Budget used exactly.

## LLM Calls

- Candidate-generation calls: 35
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 35
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 35
- Budget match: True
- LLM reflections enabled: False
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 0
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 30 / 64
- History buckets: 39
- Added candidates: 34
- Duplicate rejections: 0
- History parent selections: 10
- History offspring with history parent: 10
- Final selection from history: False

## Committee

- Output mode: committee_specialist
- Committee size: 3

### Specialists

**cand_000017** (de_curr2best_intensify)
- Status: valid
- Score: 0.49931191697497473
- Assigned instances: 0 — 
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random...

**cand_000009** (de_best_1_bin)
- Status: valid
- Score: 0.48031452318158596
- Assigned instances: 13 — 12, 13, 15, 19, 22, 23, 1, 10, 14, 2 ...
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random...

**cand_000017** (de_curr2best_intensify)
- Status: valid
- Score: 0.49931191697497473
- Assigned instances: 0 — 
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random...

### VBS (Virtual Best Solver)

- VBS mean score: 0.554486
- VBS per-instance scores: {'1': 0.9542805144809747, '10': 0.8045225409156784, '11': 0.8609766863363473, '12': 0.7129565273719745, '13': 0.6351704565449316, '14': 0.9052672173728697, '15': 0.12144946141545716, '16': 0.6139694891656415, '17': 0.7174102092312801, '18': 0.37874537124165264, '19': 0.22195815845652345, '2': 0.8987266321110001, '20': 0.22894024575333666, '21': 0.18607214749548634, '22': 0.21313505967641816, '23': 0.21882760425332012, '24': 0.12608798429250118, '3': 0.28237853649041517, '4': 0.128401121481162, '5': 0.9954761277436248, '6': 0.8659092477722832, '7': 0.5231617270692811, '8': 0.832971041307118, '9': 0.8808762555665164}

