# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000097` | valid | 10.8124 | tuned_adaptive_ils_v3 |
| 2 | `cand_000089` | valid | 10.8216 | tuned_adaptive_ils_v3 |
| 3 | `cand_000092` | valid | 10.8254 | tuned_adaptive_ils_v4 |
| 4 | `cand_000099` | valid | 10.828 | intensified_ils_v3 |
| 5 | `cand_000070` | valid | 10.8308 | adaptive_ils_tuned |

## Search Best Candidate

- ID: `cand_000097`
- Name: tuned_adaptive_ils_v3
- Status: valid
- Search Distance: 10.812403151364135
- Thought: Combines insights from parents: uses steepest 2-opt best-improvement, randomized nearest neighbor initialization (random start, greedy), adaptive perturbation (small segment reversal initially, double-bridge after stall), and increased restarts (20) and cycles (40) with stall limit 10 to improve solution quality while maintaining robustness.

## Test Evaluation

- Status: valid
- Test distance: 2.3473905494234875
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 2.3473905494234875
- Penalized mean gap: 2.3473905494234875
- Timeout penalty: 0.0
- Median gap: 2.1049677178644095
- Worst gap: 4.235748353395413
- Best gap: 1.0420575585255318

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 29
- Feedback calls: 29
- Total API calls: 134
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 29
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 17 / 30
- History buckets: 10
- Added candidates: 101
- Duplicate rejections: 1
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
