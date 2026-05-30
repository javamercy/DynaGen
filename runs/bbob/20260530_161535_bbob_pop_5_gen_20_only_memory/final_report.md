# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000049` | valid | 0.569302 | ipop_cma_local_hybrid |
| 2 | `cand_000007` | valid | 0.56869 | cma_es_restart_exploration |
| 3 | `cand_000045` | valid | 0.538339 | exploitation_focused_cma_nm |
| 4 | `cand_000029` | valid | 0.540716 | cma_intensify |
| 5 | `cand_000036` | valid | 0.526089 | exploitation_cma_nelder_local |

## Search Best Candidate

- ID: `cand_000049`
- Name: ipop_cma_local_hybrid
- Status: valid
- Search Mean AOCC: 0.5693023994851614
- Thought: This optimizer combines IPOP-CMA-ES (increasing population at each restart) with a local Nelder-Mead search after each restart. The backbone is CMA-ES with restarts for exploration; the population size doubles at each restart (IPOP) to increase global search diversity over time. After each CMA-ES run, a limited Nelder-Mead local search refines the best solution found, improving exploitation. Bounds are enforced by clipping. All randomness is controlled by the seed. The budget is carefully split among restarts and local search to avoid exceeding the limit. Initial point is random; best solution is tracked and reported.

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
- History size: 34 / 64
- History buckets: 39
- Added candidates: 49
- Duplicate rejections: 0
- History parent selections: 22
- History offspring with history parent: 20
- Final selection from history: False

## Committee

- Output mode: committee_specialist
- Committee size: 3

### Specialists

**cand_000007** (cma_es_restart_exploration)
- Status: valid
- Score: 0.5807568419420431
- Assigned instances: 7 — 21, 22, 3, 4, 19, 5, 7
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random...

**cand_000049** (ipop_cma_local_hybrid)
- Status: valid
- Score: 0.522570239877829
- Assigned instances: 5 — 12, 16, 17, 18, 15
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random...

**cand_000014** (exploit_focused_cma_es)
- Status: valid
- Score: 0.5119875624061918
- Assigned instances: 12 — 10, 1, 11, 13, 14, 2, 20, 23, 24, 6 ...
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random...

### VBS (Virtual Best Solver)

- VBS mean score: 0.602397
- VBS per-instance scores: {'1': 0.9662581782291834, '10': 0.8477418304952928, '11': 0.8605860381246309, '12': 0.8366685144249202, '13': 0.7707130060957188, '14': 0.9297724813626539, '15': 0.16306914939211178, '16': 0.8410998054895447, '17': 0.6679374757381907, '18': 0.5180958654335166, '19': 0.22900110705576704, '2': 0.8582613178027894, '20': 0.1861937741635948, '21': 0.6887101776785967, '22': 0.5038021636432346, '23': 0.2780600169261625, '24': 0.10289653821850717, '3': 0.16866058950970947, '4': 0.11875419900728826, '5': 0.981285634644498, '6': 0.9218503120018258, '7': 0.33012653327891406, '8': 0.8560573402837508, '9': 0.8319238289543873}

