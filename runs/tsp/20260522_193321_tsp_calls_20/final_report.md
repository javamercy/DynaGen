# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000012` | valid | 849.587 | multi_restart_regret2_candidate_2opt |
| 2 | `cand_000021` | valid | 848.04 | dynamic_restart_regret2_2opt |
| 3 | `cand_000027` | valid | 849.963 | dynamic_restart_regret2_2opt_reloc |

## Search Best Candidate

- ID: `cand_000012`
- Name: multi_restart_regret2_candidate_2opt
- Status: valid
- Search Distance: 849.5871450466843
- Thought: Combines multiple restarts (from best parent) with regret-2 construction (from S3 parents) and candidate-pruned 2-opt local search. Budget is distributed across restarts, with each restart using a limited number of 2-opt moves. The seed controls randomness for start city selection and candidate list permutation.

## Test Evaluation

- Status: valid
- Test distance: 5.206030788356451
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 5.206030788356451
- Penalized mean gap: 5.206030788356451
- Timeout penalty: 0.0
- Median gap: 5.337135435640804
- Worst gap: 10.363495746326373
- Best gap: 1.240484916831125

## LLM Calls

- Candidate-generation calls: 27
- Reflection calls: 7
- Feedback calls: 7
- Total API calls: 34
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 27
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 7
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
