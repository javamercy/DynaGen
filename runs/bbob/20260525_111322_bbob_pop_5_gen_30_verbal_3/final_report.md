# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000090` | valid | 0.516856 | nm_restart_de_mutation |
| 2 | `cand_000065` | valid | 0.51306 | hybrid_nelder_mead_pattern_search |
| 3 | `cand_000150` | valid | 0.510746 | latin_hypercube_initialized_nelder_mead |
| 4 | `cand_000123` | valid | 0.511031 | nm_de_archive_restart |
| 5 | `cand_000115` | valid | 0.507387 | nm_restart_anisotropic_de |

## Search Best Candidate

- ID: `cand_000065`
- Name: hybrid_nelder_mead_pattern_search
- Status: valid
- Search Mean AOCC: 0.5130600967184091
- Thought: Combines Nelder-Mead simplex with pattern search refinement. Initializes with random sampling to seed a good simplex. Nelder-Mead performs global search; when it stagnates, a short pattern search from the best point attempts local improvement. If successful, restarts simplex around the new best; otherwise random restart. Budget used efficiently through adaptive step control and limited pattern search evaluations.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.4815349148412828
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.4815349148412828
- Penalized mean AOCC: 0.4815349148412828
- Median AOCC: 0.2913152379339672
- Best AOCC: 0.9875210813168765
- Worst AOCC: 0.07006089803911988
- Mean final error: 1.9746935901392553
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.8384127294514517, 'low_moderate_conditioning': 0.652103101046173, 'multimodal_strong_global_structure': 0.18669576271570917, 'multimodal_weak_global_structure': 0.15742094615455401, 'separable': 0.607155672079504}

## LLM Calls

- Candidate-generation calls: 155
- Reflection calls: 34
- Feedback calls: 34
- Total API calls: 189
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 155
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 34
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 64 / 64
- History buckets: 39
- Added candidates: 135
- Duplicate rejections: 1
- History parent selections: 78
- History offspring with history parent: 69
- Final selection from history: False
