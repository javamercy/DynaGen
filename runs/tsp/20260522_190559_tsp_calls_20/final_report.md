# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000015` | valid | 844.161 | ils_position_mapped_2opt_tsp |
| 2 | `cand_000017` | valid | 842.357 | ils_double_bridge_tsp |
| 3 | `cand_000010` | valid | 848.244 | iterated_greedy_randomized_tsp |

## Search Best Candidate

- ID: `cand_000015`
- Name: ils_position_mapped_2opt_tsp
- Status: valid
- Search Distance: 844.1608614739558
- Thought: Recombine the Iterated Local Search (ILS) framework from cand_000010 with the O(1) position map and candidate list pruning from cand_000009. The solver uses a randomized greedy construction for initial variety, followed by a 2-opt hill climber that leverages a position map for fast node lookups. To escape local optima, it employs a 'double-bridge' style perturbation (random segment reversal) and periodic randomized restarts. The budget is managed by splitting it between the ILS loop and the local search refinement phase.

## Test Evaluation

- Status: valid
- Test distance: 5.055556784107901
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 5.055556784107901
- Penalized mean gap: 5.055556784107901
- Timeout penalty: 0.0
- Median gap: 4.165831833035195
- Worst gap: 11.078770097859445
- Best gap: 1.8821603927986907

## LLM Calls

- Candidate-generation calls: 27
- Reflection calls: 6
- Feedback calls: 6
- Total API calls: 33
- Failed calls: 8
- Main LLM model: gemma4:31b-cloud
- Feedback LLM model: gemma4:31b-cloud
- Configured candidate-generation budget: 27
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: gemma4:31b-cloud
- LLM reflections: 6
- LLM reflection errors: 0

## Archive

- Archive enabled: False
- Archive size: 0 / 64
- Archive buckets: 0
- Added candidates: 0
- Duplicate rejections: 0
- Archive parent selections: 0
- Offspring with archive parent: 0
- Final selection from archive: False
