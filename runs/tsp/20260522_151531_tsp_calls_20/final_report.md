# DynaGen Final Report

## Final Population

| Rank | Candidate     | Status | Search Distance | Name                         |
|-----:|---------------|--------|----------------:|------------------------------|
|    1 | `cand_000014` | valid  |         866.615 | regret_2opt_refined_s2       |
|    2 | `cand_000015` | valid  |         866.615 | regret_candidate_2opt_hybrid |
|    3 | `cand_000005` | valid  |         866.615 | regret_2opt_refined          |

## Search Best Candidate

- ID: `cand_000014`
- Name: regret_2opt_refined_s2
- Status: valid
- Search Distance: 866.614649930911
- Thought: Refining cand_000005 by replacing the aggressive 2-opt restart with a more systematic search. Instead of
  breaking out of both loops immediately upon finding an improvement, I will continue the inner loop to explore more
  swaps per restart, while still respecting the budget. I will also optimize the distance update logic and ensure the
  Regret-2 construction remains robust. This approach aims to explore the neighborhood more thoroughly within the budget
  constraint.

## Test Evaluation

- Status: valid
- Test distance: 6.597335262648064
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 6.597335262648064
- Penalized mean gap: 6.597335262648064
- Timeout penalty: 0.0
- Median gap: 6.709200931919447
- Worst gap: 11.12986008577656
- Best gap: 2.3130544993662863

## LLM Calls

- Candidate-generation calls: 27
- Reflection calls: 6
- Feedback calls: 6
- Total API calls: 33
- Failed calls: 12
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
