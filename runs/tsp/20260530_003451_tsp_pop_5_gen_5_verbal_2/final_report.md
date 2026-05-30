# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000007` | valid | 17.4019 | best_improve_2opt |
| 2 | `cand_000004` | valid | 17.5813 | lk_improver |
| 3 | `cand_000030` | valid | 17.5813 | nearest_2opt_limited |
| 4 | `cand_000027` | valid | 17.653 | nearest_2opt_ils |
| 5 | `cand_000024` | valid | 17.8509 | far_ins_2opt_ils |

## Search Best Candidate

- ID: `cand_000007`
- Name: best_improve_2opt
- Status: valid
- Search Distance: 17.401870528278224
- Thought: Start from nearest neighbor tour, then repeatedly apply the best 2-opt edge exchange (the one yielding maximum cost reduction) until no improving move exists. This intensifies search by making the most progress in each iteration.

## Test Evaluation

- Status: valid
- Test distance: 4.901145680488103
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 4.901145680488103
- Penalized mean gap: 4.901145680488103
- Timeout penalty: 0.0
- Median gap: 4.981251971676048
- Worst gap: 7.454068241469816
- Best gap: 1.6223067173637515

## LLM Calls

- Candidate-generation calls: 30
- Reflection calls: 6
- Feedback calls: 6
- Total API calls: 36
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 30
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 6
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 14 / 64
- History buckets: 9
- Added candidates: 25
- Duplicate rejections: 0
- History parent selections: 9
- History offspring with history parent: 9
- Final selection from history: False
