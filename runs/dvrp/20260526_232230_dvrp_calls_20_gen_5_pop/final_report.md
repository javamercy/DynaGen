# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000077` | valid | 56.9209 | min_max_return_with_depot_penalty |
| 2 | `cand_000001` | valid | 57.8334 | greedy_min_max_return |
| 3 | `cand_000065` | valid | 57.8334 | min_max_return |
| 4 | `cand_000035` | valid | 57.8334 | weighted_max_mean |
| 5 | `cand_000041` | valid | 57.8334 | weighted_max_mean |

## Search Best Candidate

- ID: `cand_000077`
- Name: min_max_return_with_depot_penalty
- Status: valid
- Search Gap: 56.92089769710084
- Thought: Select customer minimizing maximum estimated return time across all trucks, with a small penalty for customer's distance to depot to bias toward customers closer to depot. Tie-break by active truck's own return time. Returns None if no customers available.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 61.71137777103953
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 61.71137777103953
- Penalized mean gap: 61.71137777103953
- Mean TTT: 5.0253124442220996
- Penalized mean TTT: 5.0253124442220996
- Timeout penalty: 0.0
- Median gap: 64.0713395841755
- Worst gap: 122.90192923858424
- Best gap: 0.012257116269759418
- Gap by instance size: {'10': 37.95351214618832, '100': 79.46350112173084, '20': 42.6795904309726, '200': 74.9077296215167, '50': 73.55255553478924}
- TTT by instance size: {'10': 4.28783336161052, '100': 5.0518721847246235, '20': 5.852313293974405, '200': 5.016781092581803, '50': 4.917762288219146}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 24
- Feedback calls: 24
- Total API calls: 129
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 24
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 16 / 64
- History buckets: 11
- Added candidates: 56
- Duplicate rejections: 1
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
