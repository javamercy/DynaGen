# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000021` | valid | 0.507366 | ipop_cma_es |
| 2 | `cand_000031` | valid | 0.500132 | multistart_ipop_cma_es_explore |
| 3 | `cand_000028` | valid | 0.500407 | MultiStartRestartCMA |
| 4 | `cand_000023` | valid | 0.495567 | bipop_cma_es_alternate |
| 5 | `cand_000046` | valid | 0.490738 | ipop_cma_es_lhs_restart |

## Search Best Candidate

- ID: `cand_000031`
- Name: multistart_ipop_cma_es_explore
- Status: valid
- Search Mean AOCC: 0.5001316425388945
- Thought: Improves multimodal performance by restarting CMA-ES from random Latin hypercube points instead of always from the best so far, increasing exploration of different basins. Initial Latin hypercube sampling seeds a few points. Each restart doubles population size, and mean is randomly sampled from bounds (except first restart from best). This adheres to budget and seed.

## LLM Calls

- Candidate-generation calls: 50
- Reflection calls: 15
- Feedback calls: 15
- Total API calls: 65
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 125
- Budget match: False
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 15
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 37 / 64
- History buckets: 39
- Added candidates: 48
- Duplicate rejections: 0
- History parent selections: 23
- History offspring with history parent: 19
- Final selection from history: False

## Committee

- Output mode: committee_specialist
- Committee size: 3

### Specialists

**cand_000021** (ipop_cma_es)
- Status: valid
- Score: 0.4951913508685817
- Assigned instances: 14 — 10, 11, 12, 13, 14, 16, 17, 18, 19, 7 ...
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)...

**cand_000041** (exploit_intensify)
- Status: valid
- Score: 0.27848654452879346
- Assigned instances: 3 — 3, 4, 5
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func...

**cand_000015** (adaptive_coordinate_pattern_search)
- Status: valid
- Score: 0.30304122666301403
- Assigned instances: 7 — 1, 2, 21, 23, 20, 22, 6
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random...

### VBS (Virtual Best Solver)

- VBS mean score: 0.531012
- VBS per-instance scores: {'1': 0.9882194613266564, '10': 0.6498878678976787, '11': 0.6411919337313741, '12': 0.6573274713612309, '13': 0.5601802167753049, '14': 0.86854462014639, '15': 0.140442925343427, '16': 0.6888399794106728, '17': 0.4066526938855067, '18': 0.27548491228576844, '19': 0.22793367449394888, '2': 0.9754560957606684, '20': 0.21203412233759278, '21': 0.5582739628677074, '22': 0.42994085575771535, '23': 0.21271406314234192, '24': 0.09212197312851056, '3': 0.18122286263788184, '4': 0.15833293686356176, '5': 0.9985070692549163, '6': 0.9122748178684938, '7': 0.8237761066367063, '8': 0.6386851267356319, '9': 0.4462412022269104}

