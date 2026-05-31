# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000154` | timeout | 0.562089 | adaptive_runtime_lns |
| 2 | `cand_000153` | timeout | 0.671008 | AdaptiveLNS_LinearDeviation |
| 3 | `cand_000113` | timeout | 0.710507 | explorative_lns_softmax_restart |
| 4 | `cand_000152` | timeout | 0.873073 | hybrid_lns_restart_perturb |
| 5 | `cand_000161` | timeout | 0.920512 | HybridAdaptiveLNS |

## Search Best Candidate

- ID: `cand_000154`
- Name: adaptive_runtime_lns
- Status: timeout
- Search Gap: 0.5620888684442967
- Thought: Modifies the parent explorative_lns_softmax_restart by making the temperature decay adaptive based on stagnation (non-improving iterations) and adjusting removal count proportionally to stagnation. The temperature is kept high when improvements are frequent, and decays faster when stagnation persists. Removal base is 10% of customers, increased by 5% per stagnation count, capped at 40%. Restart threshold is set to 5 stagnation counts as in parent, but temperature adaptation makes restarts less frequent initially. Local search tie-breaking remains on max route distance then total distance.
- Error details: timeout on instances_000: VRP solver timed out after 180s (timeout_seconds=180)

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: -0.7317797898350255
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: -0.7317797898350255
- Penalized mean gap: -0.7317797898350255
- Mean max route distance: 2.709098284843374
- Mean total route distance: 8.772925865053647
- Timeout penalty: 0.0
- Median gap: -0.005974853637206175
- Worst gap: 4.56857841206022
- Best gap: -58.14604082612258
- Gap by instance size: {'10': -0.103839074607361, '100': 1.217122267963495, '20': -0.7095329817284262, '200': -3.7649125565424018, '50': -0.29773660426043336}
- Gap by truck count: {'1': -0.40668602816789357, '3': -0.29773660426043336, '5': 1.217122267963495, '9': -3.7649125565424018}

## LLM Calls

- Candidate-generation calls: 185
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 185
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 185
- Budget match: True
- LLM reflections enabled: False
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 0
- LLM reflection errors: 0

## History

- History enabled: False
- History size: 0 / 64
- History buckets: 0
- Added candidates: 0
- Duplicate rejections: 0
- History parent selections: 0
- History offspring with history parent: 0
- Final selection from history: False
