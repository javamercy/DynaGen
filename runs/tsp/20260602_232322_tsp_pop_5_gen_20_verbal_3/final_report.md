# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000093` | valid | 10.7746 | ils_vnd_nearest_neighbor |
| 2 | `cand_000098` | valid | 10.8025 | regret_vnd_multi50 |
| 3 | `cand_000069` | valid | 10.8107 | regret_vnd_perturb |
| 4 | `cand_000076` | valid | 10.8248 | vnd_ils_restart_40 |
| 5 | `cand_000102` | valid | 10.8336 | regret_vnd_plus |

## Search Best Candidate

- ID: `cand_000093`
- Name: ils_vnd_nearest_neighbor
- Status: valid
- Search Distance: 10.774643843709532
- Thought: Multi-start with nearest neighbor construction (random start node), then ILS: double-bridge perturbation followed by VND alternating 2-opt and Or-opt (L=1,2,3) until no improvement. 5 restarts, each with 20 ILS iterations. Reports new best tours.

## Test Evaluation

- Status: timeout
- Test distance: 2.013600259193252
- Instances evaluated: 10
- Valid runs: 6 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 4
- Mean gap: 2.013600259193252
- Penalized mean gap: 2.013600259193252
- Timeout penalty: 0.0
- Median gap: 1.9119675588019502
- Worst gap: 3.7474449239155123
- Best gap: 0.06350489276332881
- Error details: timeout on p654: Solver timed out after 600s (timeout_seconds=600)

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 26
- Feedback calls: 26
- Total API calls: 131
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 26
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 18 / 30
- History buckets: 10
- Added candidates: 97
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
