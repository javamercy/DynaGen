# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000100` | valid | 17.5833 | tighter_sa_regret |
| 2 | `cand_000098` | valid | 17.5833 | regret_diversify_aggressive |
| 3 | `cand_000096` | valid | 17.5833 | regret_2opt_swap_restart |
| 4 | `cand_000072` | valid | 17.5833 | adaptive_regret_diversify |
| 5 | `cand_000089` | valid | 17.5833 | streamlined_regret_diversify |

## Search Best Candidate

- ID: `cand_000100`
- Name: tighter_sa_regret
- Status: valid
- Search Distance: 17.583310017137904
- Thought: Regret construction with random tie-breaking; first-improvement 2-opt with delta updates; stagnation-based diversification with segment removal, random 2-opt, and random swap; simulated annealing acceptance with reduced temperature (0.05 multiplier) to limit acceptance of poor solutions; full restart after prolonged stagnation.

## Test Evaluation

- Status: valid
- Test distance: 5.788856159988621
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 5.788856159988621
- Penalized mean gap: 5.788856159988621
- Timeout penalty: 0.0
- Median gap: 5.76181733308608
- Worst gap: 9.810003877471889
- Best gap: 1.9581749049429658

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 20
- Feedback calls: 20
- Total API calls: 125
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 20
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
