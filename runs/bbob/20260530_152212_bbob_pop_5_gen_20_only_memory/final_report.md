# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000020` | valid | 0.48742 | intensified_local_search_DE |
| 2 | `cand_000033` | valid | 0.464446 | restart_intensified_DE |
| 3 | `cand_000022` | valid | 0.462375 | intensified_local_search_DE_v2 |
| 4 | `cand_000010` | valid | 0.443693 | best1_de_optimizer |
| 5 | `cand_000043` | valid | 0.431197 | enhanced_de_local_search |

## Search Best Candidate

- ID: `cand_000020`
- Name: intensified_local_search_DE
- Status: valid
- Search Mean AOCC: 0.4874195134997381
- Thought: Builds on the exploitation_intensified_DE_local parent by increasing local refinement steps and using population covariance to generate anisotropic perturbations. Maintains DE/best/1/bin for global exploration. Local refinement uses a multivariate normal centered at best with covariance estimated from the population, decaying step size, and success-based adaptation. This focuses search on promising regions while preserving diversity through regular replacement. Budget is strictly managed with early stopping on exhaustion. Seed controls all stochastic operations.

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
- History size: 33 / 64
- History buckets: 39
- Added candidates: 50
- Duplicate rejections: 0
- History parent selections: 22
- History offspring with history parent: 20
- Final selection from history: False

## Committee

- Output mode: committee_specialist
- Committee size: 3

### Specialists

**cand_000020** (intensified_local_search_DE)
- Status: valid
- Score: 0.4810792522101493
- Assigned instances: 9 — 22, 23, 11, 14, 17, 19, 2, 24, 4
- Code: import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(...

**cand_000010** (best1_de_optimizer)
- Status: valid
- Score: 0.3774095123493027
- Assigned instances: 7 — 16, 20, 18, 21, 3, 7, 9
- Code: import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3...

**cand_000033** (restart_intensified_DE)
- Status: valid
- Score: 0.4812589724973138
- Assigned instances: 8 — 1, 10, 12, 13, 6, 8, 15, 5
- Code: class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 4, ...

### VBS (Virtual Best Solver)

- VBS mean score: 0.527462
- VBS per-instance scores: {'1': 0.9211145289797922, '10': 0.7476584771663635, '11': 0.7952073119784626, '12': 0.6355643529100361, '13': 0.5795759603726899, '14': 0.8664107144185761, '15': 0.1111926089649242, '16': 0.23161579491753298, '17': 0.48035379026608394, '18': 0.46176430092296195, '19': 0.1969248755849342, '2': 0.8215985641518393, '20': 0.2354768920664556, '21': 0.5312976676997258, '22': 0.1812642604129988, '23': 0.19555584072384236, '24': 0.08859775391139603, '3': 0.16972958767664548, '4': 0.14723744934849353, '5': 0.9966719839895106, '6': 0.7850770654814068, '7': 0.8703394292137248, '8': 0.8052021247984535, '9': 0.8036634156592362}

