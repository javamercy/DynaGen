# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000096` | valid | 10.7462 | farthest_insertion_2opt_sa |
| 2 | `cand_000077` | valid | 10.7627 | improved_sa_tsp |
| 3 | `cand_000100` | valid | 10.7642 | multi_start_sa_tsp |
| 4 | `cand_000083` | valid | 10.7689 | cheapest_insertion_sa |
| 5 | `cand_000102` | valid | 10.7903 | cheapest_insertion_sa |

## Search Best Candidate

- ID: `cand_000096`
- Name: farthest_insertion_2opt_sa
- Status: valid
- Search Distance: 10.746165171885366
- Thought: Farthest insertion construction followed by steepest descent 2-opt local search, then simulated annealing with geometric cooling and 2-opt moves, yielding better initial tours and final distances.

## Test Evaluation

- Status: timeout
- Test distance: 1.8445486006577692
- Instances evaluated: 10
- Valid runs: 3 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 7
- Mean gap: 1.8445486006577692
- Penalized mean gap: 1.8445486006577692
- Timeout penalty: 0.0
- Median gap: 1.945871343219907
- Worst gap: 3.4976152623211445
- Best gap: 0.1847415062205929
- Error details: timeout on d493: Solver timed out after 600.002s (timeout_seconds=600)

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 27
- Feedback calls: 27
- Total API calls: 132
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 27
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 21 / 30
- History buckets: 10
- Added candidates: 100
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
