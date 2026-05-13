# LLM Call Comparison

Compare DynaGen, EoH, and LLaMEA as black boxes.

Use only these matching knobs:

- same search instances
- same test instances
- same LLM model
- same target LLM call count

TSP protocol:

- search instance: `synthetic:llamea:69:32`, matching LLaMEA's `generate_tsp_test(seed=69, size=32)` synthetic instance
- test instances: `data/tsp/test_instances`

Primary metric:

- `candidate_generation_calls`

Secondary checks:

- `total_api_calls`
- `failed_calls`
- `budget_match`
- `feedback_calls` when optional LLM verbal gradients are enabled; this is separate from candidate-generation calls

## DynaGen

### TSP

The TSP configs search on `synthetic:llamea:69:32` and test on `data/tsp/test_instances`.

| Target calls | Config                          | Command                                                             |
|-------------:|---------------------------------|---------------------------------------------------------------------|
|           20 | `configs/tsp/tsp_calls_20.yaml` | `python3 -m dynagen.cli run --config configs/tsp/tsp_calls_20.yaml` |
|           30 | `configs/tsp/tsp_calls_30.yaml` | `python3 -m dynagen.cli run --config configs/tsp/tsp_calls_30.yaml` |
|           40 | `configs/tsp/tsp_calls_40.yaml` | `python3 -m dynagen.cli run --config configs/tsp/tsp_calls_40.yaml` |

### BBOB

| Target calls | Config                            | Command                                                               |
|-------------:|-----------------------------------|-----------------------------------------------------------------------|
|           20 | `configs/bbob/bbob_calls_20.yaml` | `python3 -m dynagen.cli run --config configs/bbob/bbob_calls_20.yaml` |
|           30 | `configs/bbob/bbob_calls_30.yaml` | `python3 -m dynagen.cli run --config configs/bbob/bbob_calls_30.yaml` |
|           40 | `configs/bbob/bbob_calls_40.yaml` | `python3 -m dynagen.cli run --config configs/bbob/bbob_calls_40.yaml` |

## LLaMEA

### TSP

The native multi-objective TSP example searches on `generate_tsp_test(seed=69, size=32)` and writes a final
`test_result.json` from `data/tsp/test_instances` by default.

| Target calls | Env vars                                                                                                                                  | Command                                                                                                                                                                                                     |
|-------------:|-------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|           20 | `LLM_MODEL=... LLAMEA_LLM_CALLS=20 LLAMEA_TSP_SEARCH_SEED=69 LLAMEA_TSP_SEARCH_SIZE=32 LLAMEA_TSP_TEST_INSTANCES=data/tsp/test_instances` | `LLM_MODEL=... LLAMEA_LLM_CALLS=20 LLAMEA_TSP_SEARCH_SEED=69 LLAMEA_TSP_SEARCH_SIZE=32 LLAMEA_TSP_TEST_INSTANCES=data/tsp/test_instances uv run python examples/multi_objective.py` from `baselines/LLaMEA` |
|           30 | `LLM_MODEL=... LLAMEA_LLM_CALLS=30 LLAMEA_TSP_SEARCH_SEED=69 LLAMEA_TSP_SEARCH_SIZE=32 LLAMEA_TSP_TEST_INSTANCES=data/tsp/test_instances` | `LLM_MODEL=... LLAMEA_LLM_CALLS=30 LLAMEA_TSP_SEARCH_SEED=69 LLAMEA_TSP_SEARCH_SIZE=32 LLAMEA_TSP_TEST_INSTANCES=data/tsp/test_instances uv run python examples/multi_objective.py` from `baselines/LLaMEA` |
|           40 | `LLM_MODEL=... LLAMEA_LLM_CALLS=40 LLAMEA_TSP_SEARCH_SEED=69 LLAMEA_TSP_SEARCH_SIZE=32 LLAMEA_TSP_TEST_INSTANCES=data/tsp/test_instances` | `LLM_MODEL=... LLAMEA_LLM_CALLS=40 LLAMEA_TSP_SEARCH_SEED=69 LLAMEA_TSP_SEARCH_SIZE=32 LLAMEA_TSP_TEST_INSTANCES=data/tsp/test_instances uv run python examples/multi_objective.py` from `baselines/LLaMEA` |

### BBOB

Run LLaMEA through its native example scripts. The scripts write LLaMEA's own experiment logs under `runs/bbob` by
default.

Set `LLAMEA_LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` to match DynaGen's OpenAI setup.
Current LLaMEA BBOB uses a single evaluation pool, so it matches DynaGen's search-side shape but does not yet
mirror DynaGen's separate held-out `test_instances` and `test_dimensions`.

#### LLaMEA

| Target calls | Env vars                            | Command                                                                                                      |
|-------------:|-------------------------------------|--------------------------------------------------------------------------------------------------------------|
|           20 | `LLM_MODEL=... LLAMEA_LLM_CALLS=20` | `LLM_MODEL=... LLAMEA_LLM_CALLS=20 uv run python examples/black-box-optimization.py` from `baselines/LLaMEA` |
|           30 | `LLM_MODEL=... LLAMEA_LLM_CALLS=30` | `LLM_MODEL=... LLAMEA_LLM_CALLS=30 uv run python examples/black-box-optimization.py` from `baselines/LLaMEA` |
|           40 | `LLM_MODEL=... LLAMEA_LLM_CALLS=40` | `LLM_MODEL=... LLAMEA_LLM_CALLS=40 uv run python examples/black-box-optimization.py` from `baselines/LLaMEA` |

#### LLaMEA-HPO

| Target calls | Env vars                            | Command                                                                                                      |
|-------------:|-------------------------------------|--------------------------------------------------------------------------------------------------------------|
|           20 | `LLM_MODEL=... LLAMEA_LLM_CALLS=20` | `LLM_MODEL=... LLAMEA_LLM_CALLS=20 uv run python examples/black-box-opt-with-HPO.py` from `baselines/LLaMEA` |
|           30 | `LLM_MODEL=... LLAMEA_LLM_CALLS=30` | `LLM_MODEL=... LLAMEA_LLM_CALLS=30 uv run python examples/black-box-opt-with-HPO.py` from `baselines/LLaMEA` |
|           40 | `LLM_MODEL=... LLAMEA_LLM_CALLS=40` | `LLM_MODEL=... LLAMEA_LLM_CALLS=40 uv run python examples/black-box-opt-with-HPO.py` from `baselines/LLaMEA` |

Configurable LLaMEA BBOB run knobs are environment variables in `baselines/LLaMEA/examples/black-box-optimization.py`
and `baselines/LLaMEA/examples/black-box-opt-with-HPO.py`:

- `LLAMEA_OUTPUT_DIR`
- `LLAMEA_LLM_MODEL`
- `LLAMEA_LLM_CALLS`
- `LLAMEA_N_PARENTS`
- `LLAMEA_N_OFFSPRING`
- `LLAMEA_BBOB_DIMENSIONS`
- `LLAMEA_BBOB_FUNCTION_IDS`
- `LLAMEA_BBOB_INSTANCE_IDS`
- `LLAMEA_BBOB_REPETITIONS`
- `LLAMEA_BBOB_BUDGET_FACTOR`
- `LLAMEA_HPO_TRIALS` for LLaMEA-HPO

Configurable LLaMEA TSP instance knobs are environment variables in `baselines/LLaMEA/examples/multi_objective.py`:

- `LLAMEA_TSP_SEARCH_SEED`
- `LLAMEA_TSP_SEARCH_SIZE`
- `LLAMEA_TSP_TEST_INSTANCES`

## EoH

### TSP

| Target calls | Env vars                                                                                                                                        | Command                                                                                                                                                                                                                           |
|-------------:|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|           20 | `LLM_MODEL=... EOH_TSP_SEARCH_INSTANCES=synthetic:llamea:69:32 EOH_TSP_TEST_INSTANCES=data/tsp/test_instances EOH_EC_POP_SIZE=2 EOH_EC_N_POP=2` | `LLM_MODEL=... EOH_TSP_SEARCH_INSTANCES=synthetic:llamea:69:32 EOH_TSP_TEST_INSTANCES=data/tsp/test_instances EOH_EC_POP_SIZE=2 EOH_EC_N_POP=2 OPENAI_API_KEY=... python3 baselines/EoH/examples/tsp_construct/runEoH_compare.py` |
|           30 | `LLM_MODEL=... EOH_TSP_SEARCH_INSTANCES=synthetic:llamea:69:32 EOH_TSP_TEST_INSTANCES=data/tsp/test_instances EOH_EC_POP_SIZE=1 EOH_EC_N_POP=7` | `LLM_MODEL=... EOH_TSP_SEARCH_INSTANCES=synthetic:llamea:69:32 EOH_TSP_TEST_INSTANCES=data/tsp/test_instances EOH_EC_POP_SIZE=1 EOH_EC_N_POP=7 OPENAI_API_KEY=... python3 baselines/EoH/examples/tsp_construct/runEoH_compare.py` |
|           40 | `LLM_MODEL=... EOH_TSP_SEARCH_INSTANCES=synthetic:llamea:69:32 EOH_TSP_TEST_INSTANCES=data/tsp/test_instances EOH_EC_POP_SIZE=4 EOH_EC_N_POP=2` | `LLM_MODEL=... EOH_TSP_SEARCH_INSTANCES=synthetic:llamea:69:32 EOH_TSP_TEST_INSTANCES=data/tsp/test_instances EOH_EC_POP_SIZE=4 EOH_EC_N_POP=2 OPENAI_API_KEY=... python3 baselines/EoH/examples/tsp_construct/runEoH_compare.py` |

## Outputs

Compare these artifacts from each run:

- `llm_calls.json`
- `test_result.json`

Include the `model` field from `llm_calls.json` and the `llm_model` field from `test_result.json` when reporting
results.
