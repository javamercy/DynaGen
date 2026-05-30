# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000018` | valid | 17.4757 | iterated_local_search |
| 2 | `cand_000016` | valid | 17.5813 | nearest_neighbor_2opt |
| 3 | `cand_000024` | timeout | 17.5939 | fast_regret_2opt_doublebridge_2 |
| 4 | `cand_000011` | timeout | 17.5995 | regret_2opt |
| 5 | `cand_000014` | timeout | 17.6132 | perturbed_2opt_regret |

## Search Best Candidate

- ID: `cand_000018`
- Name: iterated_local_search
- Status: valid
- Search Distance: 17.475689440182276
- Thought: Start with regret insertion (farthest pair initialization, then insert city with maximum regret). Improve with 2-opt until local optimum. Then apply a double-bridge perturbation (swap two random disjoint segments) to escape local minima, followed by 2-opt again. Repeat for a fixed number of iterations (e.g., 20 times or until timeout). Always track and report the best tour found. This combines constructive heuristic with iterated local search for better solutions.

## Test Evaluation

- Status: timeout
- Test distance: 3.8169047255887505
- Instances evaluated: 10
- Valid runs: 9 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 1
- Mean gap: 3.8169047255887505
- Penalized mean gap: 3.8169047255887505
- Timeout penalty: 0.0
- Median gap: 3.876985386527248
- Worst gap: 6.404724051782876
- Best gap: 1.0519645120405576
- Error details: timeout on rat783: Solver timed out after 60.0008s (timeout_seconds=60)

## LLM Calls

- Candidate-generation calls: 30
- Reflection calls: 8
- Feedback calls: 8
- Total API calls: 38
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 30
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 8
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 13 / 64
- History buckets: 9
- Added candidates: 28
- Duplicate rejections: 0
- History parent selections: 9
- History offspring with history parent: 9
- Final selection from history: False
