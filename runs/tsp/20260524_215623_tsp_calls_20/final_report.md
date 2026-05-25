# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000010` | valid | 17.5547 | farthest_regret_hybrid_restart |
| 2 | `cand_000008` | valid | 17.5602 | farthest_regret_hybrid |
| 3 | `cand_000007` | valid | 17.5958 | ils_double_bridge |

## Search Best Candidate

- ID: `cand_000010`
- Name: farthest_regret_hybrid_restart
- Status: valid
- Search Distance: 17.5546819015446
- Thought: Construct tour using farthest pair initialization and regret-based insertion with tie-breaking by farthest insertion cost, then apply 2-opt local search with restart after each improvement, all budget-bounded.

## Test Evaluation

- Status: valid
- Test distance: 5.534005482668505
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 5.534005482668505
- Penalized mean gap: 5.534005482668505
- Timeout penalty: 0.0
- Median gap: 5.979197695740995
- Worst gap: 8.76308646762311
- Best gap: 1.8631178707224334

## LLM Calls

- Candidate-generation calls: 12
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 12
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 12
- Budget match: True
- LLM reflections enabled: False
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 0
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
