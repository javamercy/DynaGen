# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000026` | valid | 1676.96 | regret_two_construction |
| 2 | `cand_000027` | valid | 1692.86 | regret2_2opt |
| 3 | `cand_000025` | valid | 1741.24 | farthest_insertion_2opt_candidate_list |

## Search Best Candidate

- ID: `cand_000026`
- Name: regret_two_construction
- Status: valid
- Search Distance: 1676.9642431216062
- Thought: Replace random insertion with regret-2 construction to generate higher-quality initial tours, while keeping first-improvement 2-opt with candidate lists and restart-on-plateau. Budget counts each delta evaluation and restart.

## Test Evaluation

- Status: valid
- Test distance: 6.26527545533313
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 6.26527545533313
- Penalized mean gap: 6.26527545533313
- Timeout penalty: 0.0
- Median gap: 6.744350769538103
- Worst gap: 10.845110033045067
- Best gap: 2.2836199605300256

## LLM Calls

- Candidate-generation calls: 27
- Reflection calls: 8
- Feedback calls: 8
- Total API calls: 35
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 27
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 8
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
