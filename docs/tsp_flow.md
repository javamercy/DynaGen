# DynaGen TSP Flow

This document describes the current TSP flow implemented in DynaGen. It follows the actual code path used by
`python3 -m dynagen.cli run --config configs/tsp/*.yaml`.

## 1. Command Entry

1. The user starts a run with `python3 -m dynagen.cli run --config configs/tsp/tsp_calls_20.yaml` or another TSP config.
2. `dynagen.cli.main` parses the `run` subcommand and requires `--config`.
3. `load_config` reads the config file as JSON, TOML, YAML, or YML.
4. The TSP configs do not need an explicit `problem.type` field because `ProblemConfig.type` defaults to `tsp`.
5. `RunConfig.from_dict` builds these sections:

| Section      | Dataclass          | Important TSP fields                                                     |
|--------------|--------------------|--------------------------------------------------------------------------|
| `run`        | `RunConfig`        | `name`,  `output_dir`, `seed`                                            |
| `llm`        | `LLMConfig`        | `provider`, `model`, `temperature`, `api_key_env`                        |
| `evolution`  | `EvolutionConfig`  | `population_size`, `generations`, `offspring_per_strategy`, `strategies`, `verbal_gradients`, `archive` |
| `evaluation` | `EvaluationConfig` | `budget`, `timeout_seconds`, `timeout_penalty`, `seeds`, `metric`        |
| `problem`    | `ProblemConfig`    | defaults to `type: tsp` unless overridden                                |
| `data`       | `DataConfig`       | `search_instances`, `test_instances`                                     |

6. `EvaluationConfig` requires at least one seed, a metric, a positive budget, and positive timeout seconds.
7. `EvolutionConfig` converts strategy names into the `Strategy` enum. If strategies are not specified, it uses all
   strategies: `S1`, `S2`, and `S3`.
8. `scheduled_llm_calls` computes the configured candidate-generation call budget as
   `population_size + generations * len(strategies) * offspring_per_strategy`.
9. `evolution.verbal_gradients` can independently enable static feedback, LLM feedback cadence, and the LLM model used
   for feedback calls.
10. `evolution.archive` controls quality-diversity memory, archive parent sampling, final archive selection, and archive
    persistence.

## 2. LLM Provider Setup

1. `_provider_from_config` creates the concrete provider.
2. If `llm.provider` is `openai`, `OpenAIProvider` reads the API key from `llm.api_key_env`, creates an OpenAI client,
   and uses chat completions.
3. OpenAI responses are requested with a strict JSON schema named `candidate_response`.
4. If `llm.provider` starts with `ollama`, `OllamaProvider` sends a non-streaming request to
   `http://localhost:11434/api/chat` with the same candidate response schema as the requested format.
5. The concrete provider is wrapped in `CountingLLMProvider`.
6. `CountingLLMProvider` increments `candidate_generation_calls` and `total_api_calls` before each candidate-generation
   request.
7. If a provider call raises, `CountingLLMProvider` increments `failed_calls` and re-raises the exception.
8. `complete_text` calls are used for optional verbal-gradient feedback. They increment `feedback_calls` and the
   compatibility alias `reflection_calls`, but they do not increment `candidate_generation_calls`.
9. `evolution.verbal_gradients.llm_every_n_generations` controls how often LLM feedback is generated, and
   `evolution.verbal_gradients.llm_model` can override the model used for those feedback calls.
10. The provider summary is later written to `llm_calls.json`.

## 3. Problem And Evaluator Setup

1. `_build_evaluator(config, pool_name="search_instances")` calls `problem_for_config(config).build_evaluator`.
2. `problem_for_config` returns `TSPProblem` when `problem.type` is `tsp`.
3. `TSPProblem.build_evaluator` chooses the data path from `config.data.search_instances` for the search pool and
   `config.data.test_instances` for the test pool.
4. `load_tsp_instances` rejects missing paths.
5. If a path matches `synthetic:llamea:seed:size`, `parse_llamea_tsp_specs` extracts one synthetic instance.
   If it matches `synthetic:llamea:seeds=11,23,69:sizes=32,50,100,200`, it expands the seed-size grid.
6. Synthetic TSP generation starts with a depot at `(50.0, 50.0)` and adds `size` random points with
   `random.Random(seed)`.
7. Synthetic distances are exact Euclidean distances, the diagonal is set to zero, and `optimal_length` is `None`.
8. Synthetic metadata includes `source`, `generator`, `seed`, `customer_count`, and `depot`.
9. Current TSP configs use the 12-instance synthetic search pool from seeds `11`, `23`, and `69` crossed with sizes
   `32`, `50`, `100`, and `200`.
10. If a path is a directory, every `.tsp` file in sorted order is loaded.
11. If a path is a file, that single file is loaded.
12. TSPLIB parsing supports `TYPE: TSP` only.
13. TSPLIB coordinate instances support `EDGE_WEIGHT_TYPE: EUC_2D` and `CEIL_2D`.
14. TSPLIB explicit matrix instances support `EDGE_WEIGHT_TYPE: EXPLICIT` with a full square matrix.
15. `OPTIMAL` headers are parsed into `optimal_length` when present.
16. `TSPInstance` validates that the distance matrix is square, finite, non-negative, has a zero diagonal, and is
    symmetric.
17. Coordinates, when present, must have one row per node and at least two columns.
18. Two `TSPCandidateEvaluator` instances are created: one for search evaluation and one for final test evaluation.
19. Each evaluator stores instances, seeds, budget, timeout seconds, timeout penalty, and pool name.

## 4. Run Store Setup

1. `RunStore.create` creates a timestamped run directory under `run.output_dir`.
2. The run directory name is `YYYYMMDD_HHMMSS_<safe_run_name>`.
3. If the computed directory already exists, suffixes like `_01` are tried until a free directory is found.
4. `config.json` is written immediately with the resolved config.
5. These subdirectories are created:

| Directory      | Purpose                                                |
|----------------|--------------------------------------------------------|
| `candidates/`  | Candidate metadata JSON and candidate Python source    |
| `prompts/`     | Full prompt text, and response text if available       |
| `generations/` | Per-generation population, offspring, and summary JSON |
| `feedback/`    | Optional LLM verbal-gradient prompts and responses      |
| `archive/`     | Current archive state and per-generation archive snapshots |

6. Candidate IDs are generated as `cand_000001`, `cand_000002`, and so on.
7. Candidate ID generation is protected by a lock because candidate tasks run in parallel.

## 5. Evolution Engine Initialization

1. `EvolutionEngine` receives the config, counting provider, search evaluator, test evaluator, and store.
2. The engine resolves the problem again with `problem_for_config(config)`.
3. The engine creates `random.Random(config.seed)` for deterministic parent selection.
4. `EvolutionEngine.run` is the main orchestration method.

## 6. Initial Population Construction

1. `_initial_population` asks `TSPProblem.initial_roles(population_size)` for initial TSP roles.
2. `create_tsp_initial_roles` cycles over `TSP_INITIAL_ROLES` if `population_size` exceeds the number of predefined
   roles.
3. Each role has a slot number, role text, and intended algorithmic bias.
4. Current initial role themes are size-adaptive candidate-list local search, regret/cheapest insertion construction,
   iterated local-search diversification, large-instance scalability, and small/medium intensive refinement.
5. `_build_initial_tasks` creates one `_CandidateTask` per role.
6. For each task, `TSPProblem.build_initial_prompt` calls `build_tsp_initial_prompt`.
7. The system prompt is built by `tsp_system_prompt(role.role)`.
8. The user prompt includes the assigned solver perspective, behavioral bias, `TSP_SOLVER_CONTRACT`,
   `TSP_INTERNAL_CHECKLIST`, and `TSP_RESPONSE_FORMAT`.
9. `_format_messages` serializes the message list into a stored prompt string with `[system]` and `[user]` sections.
10. Each initial task has generation `0`, strategy `initial:<slot>`, no parents, a candidate ID, raw messages, and
    stored prompt text.

## 7. Candidate Task Execution

1. `_execute_tasks_parallel` processes candidate tasks with a `ThreadPoolExecutor`.
2. The worker count is `min(number_of_tasks, 8)`.
3. Tasks may complete out of order, but results are stored back into the original task order.
4. `_process_single_task` is responsible for one LLM call, one search evaluation, and persistence.
5. `provider.complete(task.messages, temperature=config.llm.temperature)` sends the prompt to the provider.
6. The provider returns `ParsedCandidateResponse` with `name`, `thought`, and `code`.
7. If JSON parsing, schema handling, provider execution, or any later step raises before a candidate exists,
   `_failed_candidate` creates an error candidate.
8. A failed candidate has empty `name`, `thought`, and `code`, status `error`, distance `inf`, empty metrics from the
    evaluator, prompt text, parents, and `error_details`.
9. If a candidate already exists and evaluation raises unexpectedly, `_mark_candidate_error` sets status `error`,
   distance `inf`, and stores the error details.
10. Successful parsing creates a `Candidate` with status `pending`, no distance yet, empty metrics, the returned code,
    prompt text, generation, strategy, and parent IDs.
11. The search evaluator mutates the candidate with status, distance, metrics, and error details.
12. Every candidate is saved even if it is invalid, timed out, or errored.

## 8. Candidate Response Contract

1. The LLM must return exactly one JSON object.
2. The object must contain `name`, `thought`, and `code`.
3. `name` is a short solver name.
4. `thought` is a public high-level summary of the algorithmic idea.
5. `code` is complete Python source code as a JSON string.
6. The code must define `solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int)`.
7. The solver must return a one-dimensional tour containing every node ID exactly once.
8. The solver must not repeat the start node at the end; the evaluator closes the cycle.
9. The solver must work across TSP sizes and return trivial tours for `n <= 2`.
10. The solver should create and report a valid incumbent early.
11. The solver must use `seed` for stochastic behavior when randomness is used.
12. The solver must treat `budget` as a hard effort cap.
13. The solver must not hard-code benchmark instances, matrix sizes, seeds, or evaluator artifacts.
14. The solver must use the distance matrix directly and cannot assume coordinates are available.
15. The solver must not read files, write files, access the network, spawn subprocesses, or call external solvers.
16. Allowed imports are `numpy`, `math`, `random`, `heapq`, `itertools`, `collections`, and `time`.
17. `report_best_tour(tour)` is available globally inside the sandbox.
18. The solver should call `report_best_tour` for the initial incumbent and every improved incumbent.
19. Reporting does not replace returning a final tour.

## 9. Static Code Validation

1. `TSPCandidateEvaluator.evaluate_code` first calls `validate_generated_code`.
2. Validation parses the candidate source with `ast.parse`.
3. Syntax errors make the candidate invalid before execution.
4. Only these top-level statements are allowed for TSP: imports, function definitions, assignments, annotated
   assignments, and constant expressions such as docstrings.
5. Every import is checked against the allowed import roots.
6. Relative imports are rejected.
7. Unsafe direct calls such as `open`, `exec`, `eval`, `compile`, `input`, `__import__`, `globals`, `locals`, and `vars`
   are rejected.
8. Unsafe attribute calls such as `load`, `save`, `savetxt`, `loadtxt`, `genfromtxt`, `fromfile`, `tofile`, `memmap`,
   `system`, `popen`, `spawn`, `fork`, `unlink`, `rmdir`, and `mkdir` are rejected.
9. A function named `solve_tsp` must exist.
10. The AST signature of `solve_tsp` must have exactly three positional parameters named `distance_matrix`, `seed`, and
    `budget`.
11. The code is compiled after AST validation.
12. If validation fails, no solver process is launched.
13. Invalid validation returns an `EvaluationResult` with status `invalid`, distance `inf`, empty TSP metrics with
    context, and the validation error text.

## 10. Search Evaluation Execution

1. If static validation succeeds, `_run_all_instances` builds all `(instance, seed)` tasks.
2. The number of evaluation tasks is `len(instances) * len(seeds)`.
3. Evaluation tasks run in a `ThreadPoolExecutor` with up to 8 workers.
4. Each task calls `_run_single_instance`.
5. `_run_single_instance` calls `run_tsp_solver(code, instance, seed, budget, timeout_seconds)`.
6. `run_tsp_solver` delegates to `execute_tsp_solver_code` with the instance distance matrix.
7. The candidate receives a copy of the distance matrix, not the original evaluator matrix.
8. The same evaluator budget is passed to every instance and seed run.
9. The same timeout seconds value is used as a process wall-clock cap for every run.

## 11. Sandboxed Solver Process

1. `execute_tsp_solver_code` creates a multiprocessing context, preferring `spawn`, then `forkserver`, then the default
   context.
2. The distance matrix is converted to a float NumPy array.
3. The matrix dimension determines the size of two shared arrays for reported tours.
4. Two shared arrays are used so a reported incumbent can be copied without exposing a half-written active tour.
5. `active_tour_index` starts at `-1`, meaning no tour has been reported yet.
6. A result queue with `maxsize=1` receives the solver process result.
7. A child process runs `_worker`.
8. The parent starts the process, waits with `process.join(timeout_seconds)`, and measures runtime with
   `time.perf_counter`.
9. If the process is still alive after the timeout, the parent terminates it.
10. If termination does not stop the process after one second, the parent kills it.
11. On timeout, the parent reads the latest reported tour from the shared arrays if one exists.
12. Inside `_worker`, `report_best_tour` converts the reported tour to a flat float array.
13. `report_best_tour` ignores reported tours whose length does not match the TSP dimension.
14. `report_best_tour` writes the reported tour into the inactive shared array and then flips `active_tour_index`.
15. `load_tsp_solver` validates static code again by default before executing it.
16. The sandbox namespace includes safe builtins, `np`, `numpy`, allowed modules, `report_best_tour`, and a no-op
    `report_best`.
17. The sandbox replaces `__import__` with `_safe_import`, which only allows configured safe modules.
18. The generated source is compiled and executed in the sandbox namespace.
19. The sandbox retrieves `solve_tsp` and verifies it is callable.
20. The worker calls `solve_tsp(distance_matrix.copy(), int(seed), int(budget))`.
21. On normal return, the worker converts the returned tour to a list and puts status `ok`, the value, runtime, and no
    error into the result queue.
22. On exception, the worker puts status `error`, no value, runtime, and a short error message into the result queue.
23. If the child exits without writing to the queue, the parent records an error and includes any reported tour if one
    exists.

## 12. TSP Run Result Handling

1. `run_tsp_solver` receives `TSPSolverExecutionResult` from the process layer.
2. If the execution status is `timeout` and a reported tour exists, the reported tour is validated.
3. A valid reported tour on timeout returns `TSPSolverRunResult` with status `timeout`, `partial=True`, tour, tour
   length, runtime, and timeout error text.
4. An invalid reported tour on timeout returns status `timeout` without a scoreable tour and includes the validation
   failure in the error text.
5. A timeout without any reported tour returns status `timeout` with no tour and no tour length.
6. A non-OK execution status other than timeout returns status `error`.
7. An OK execution validates the returned final tour.
8. A valid returned tour returns status `valid`, the normalized tour, and its tour length.
9. An invalid returned tour returns status `invalid`, no tour length, and a short validation error.
10. Tour validation is handled by `TSPInstance.validate_tour` and `dynagen.domain.tour.validate_tour`.
11. A valid tour must be one-dimensional.
12. A valid tour must have exactly `dimension` entries.
13. Tour entries must be integer node IDs or numeric values exactly equal to integers.
14. Node IDs must be in the range `[0, dimension - 1]`.
15. Every node must appear exactly once.
16. Tour length is computed by summing `distance_matrix[nodes, np.roll(nodes, -1)]`, which closes the cycle back to the
    first node.

## 13. Per-Run Record Construction

1. `_run_single_instance` marks a run as scored when its status is `valid` or it is a partial timeout.
2. If the run is scored and has a tour length, `compute_gap` is called.
3. `compute_gap` requires a positive finite tour length.
4. If the instance has no `optimal_length`, the gap is `None`.
5. If the instance has a valid positive `optimal_length`, the gap is
   `100.0 * (tour_length - optimal_length) / optimal_length`.
6. `reference_kind` is `optimal` when `optimal_length` exists, otherwise `None`.
7. Each record contains instance name, pool name, dimension, source, seed, status, tour length, partial flag, reference
   length, reference kind, gap, runtime seconds, and error text.

## 14. Metric Aggregation

1. `aggregate_tsp_records` receives every per-run record for a candidate.
2. `valid_count` counts records whose status is `valid`.
3. `scored_count` counts records with a finite gap.
4. `timeout_count` counts timeout records.
5. `partial_timeout_count` counts timeout records that still have a finite gap.
6. `unscored_timeout_count` counts timeout records without a finite gap.
7. `invalid_tour_count` counts records with status `invalid`.
8. `runtime_error_count` counts records with status `error`.
9. `mean_tour_length` is the mean of all finite tour lengths, regardless of whether a gap exists.
10. `mean_gap`, `median_gap`, `best_gap`, and `worst_gap` are computed only from finite gaps.
11. `timeout_fraction` is `timeout_count / runs`.
12. `penalized_mean_gap` is `mean_gap + timeout_penalty * timeout_fraction` when `mean_gap` exists.
13. `timeout_distance` is `penalized_mean_gap` when it exists.
14. If timeout records exist but no finite penalized gap exists, `timeout_distance` is `1_000_000.0`.
15. `mean_runtime` is the mean runtime across records.
16. `score_by_instance_size` groups lower-is-better primary scores by `dimension`: gap when available, otherwise tour
   length.
17. `score_by_instance_source` groups the same lower-is-better primary scores by source path or source string.
18. `gap_by_instance_size`, `gap_by_instance_source`, `tour_length_by_instance_size`, and
   `tour_length_by_instance_source` keep explicit grouped metric views.
19. The full records list is stored under `records`.
20. `TSPCandidateEvaluator._with_context` adds `pool`, `seeds`, `budget`, and `timeout_penalty` to the metrics.

## 15. Candidate Status And Distance

1. `_candidate_status` maps aggregate metrics to a candidate-level status.
2. If any timeout exists, the candidate status is `timeout`.
3. If no timeout exists but any runtime error exists, the candidate status is `error`.
4. If no timeout or runtime error exists but any invalid tour exists, the candidate status is `invalid`.
5. If all runs are valid, the candidate status is `valid`.
6. Any other mixed state is treated as `invalid`.
7. `_candidate_distance` maps status and metrics to a lower-is-better distance value.
8. A valid search-pool TSP candidate uses `mean_tour_length` as distance.
9. This matters for the synthetic LLaMEA search pool because its instances have no known optimum and therefore no gap.
10. A valid test-pool TSP candidate uses `mean_gap` when available.
11. If a valid test-pool candidate has no mean gap, it falls back to `mean_tour_length`.
12. A timeout candidate uses `timeout_distance` when available.
13. Invalid and error candidates receive distance `inf`.
14. `evaluate_candidate` writes the status, distance, metrics, and error feedback back onto the `Candidate` object.
15. Error feedback for non-valid evaluations reports the first non-valid record as
    `<status> on <instance> seed <seed>: <message>`.

## 16. Candidate Verbal Gradients

1. After every candidate evaluation or failure, the engine attaches a static evaluator-derived verbal gradient when
   `evolution.verbal_gradients.enabled` and `static_enabled` are true.
2. The gradient is stored at `candidate.metrics["verbal_gradient"]`.
3. The TSP static gradient summarizes status, distance, parent delta, mean/worst/best gaps when available, timeout
   fraction, runtime, score by size/source, and error details.
4. The gradient contains a summary, mechanisms to preserve, measured weaknesses, strategy-specific next mutations for
   `S1`, `S2`, and `S3`, and behaviors to avoid.
5. Before each offspring prompt is built, the engine ensures the selected parents have gradients.
6. If `llm_enabled` is true, selected parents without an LLM gradient can receive a cached LLM verbal gradient, capped by
   `max_llm_calls_per_generation`.
7. LLM gradient calls use `complete_text`, write records to `feedback/`, and do not count against candidate-generation
   budget.
8. If LLM gradient parsing or generation fails, the candidate keeps its static gradient and records the LLM error.

## 17. Candidate Persistence

1. `RunStore.save_candidate` writes candidate metadata to `candidates/<candidate_id>.json` without embedding the code.
2. The candidate source code is written to `candidates/<candidate_id>.py`.
3. If prompt text exists, it is written to `prompts/<candidate_id>_prompt.txt`.
4. If raw response text exists, it is written to `prompts/<candidate_id>_response.txt`.
5. Candidate JSON includes ID, generation, strategy, name, thought, parents, distance, metrics, prompt text, raw
    response, error details, status, and creation time.

## 18. Generation 0 Survivor Selection

1. `_initial_population` registers all generated candidates and updates the archive before survivor selection.
2. `_initial_population` returns `Population.from_candidates(0, candidates, size=population_size)`.
3. `Population.from_candidates` calls `select_survivors`.
4. `select_survivors` uses metric-vector ranking: usable status class, scalar score band, worst group score,
   timeout fraction, validity, status penalty, runtime, novelty, raw scalar score, generation, and candidate ID.
5. `valid`, `evaluated`, and `timeout` candidates share the usable status class, so a scoreable timeout can beat a
   valid candidate when its scalar score is materially better.
6. Only the first `population_size` candidates survive.
7. `EvolutionEngine.run` saves `generation_000` immediately after the initial population is selected.
8. `generation_000/population.json` contains survivor metadata.
9. `generation_000/offspring.json` is empty for generation 0.
10. `generation_000/summary.json` contains generation number, population IDs, offspring count, valid offspring count,
    best candidate ID, best distance, and status counts.
11. `archive/generation_000.json` stores the archive snapshot after generation 0 candidates have been profiled.

## 19. Offspring Generation Loop

1. The engine loops from generation `1` through `config.evolution.generations`, inclusive.
2. `_generate_offspring` calls `_build_offspring_tasks` for the current population.
3. For each configured strategy and for each `offspring_per_strategy`, one task is created.
4. Default TSP strategies are `S1`, `S2`, and `S3`.
5. `parent_count(S1)` is `1`.
6. `parent_count(S2)` is `1`.
7. `parent_count(S3)` is `3`.
8. `_select_strategy_parents` uses the engine RNG.
9. Parent selection first filters current-population candidates with status `valid`, `evaluated`, or `timeout`.
10. If archive is enabled, each parent slot can be sampled from the archive according to
    `archive.parent_sample_probability`.
11. For S3, `archive.s3_archive_parent_min` can force at least one archived parent when available.
12. If the sampled source is empty, selection falls back to the other source.
13. If no eligible current-population candidates exist, all population candidates are eligible.
14. Current-population parent probabilities are rank-biased using the same survivor sort key.
15. Archive parent probabilities are rank-biased by archive score order.
16. A current-population candidate at rank `r` in a population of size `n` receives unnormalized weight
    `1.0 / (r + n)`.
17. Weights are normalized over the current pool.
18. Parent selection is without replacement for a single task because a selected parent is removed from the temporary
    pool.
19. Archive-selected parents receive a compact parent-context marker with archive source, bucket, and role.
20. The engine formats only the selected parents' full verbal gradients into a parent-specific feedback block.
21. Optional LLM gradients are cached on parent candidates, so repeated parent use does not require repeated feedback
    calls.
22. Offspring generation does not use the old generation-wide reflection broadcast.

## 20. TSP Evolution Prompts

1. `TSPProblem.build_evolution_prompt` calls `build_tsp_evolution_prompt`.
2. The prompt rejects unknown strategy names.
3. Parent context is rendered with candidate ID, name, status, distance, thought, optional error details, optional archive
   source marker, and full code.
4. Strategy `S1` is explorative innovation.
5. `S1` asks for a meaningfully different algorithmic approach from the selected parent.
6. `S1` suggests changes to construction heuristics, local search neighborhoods, perturbation mechanisms, acceptance
   criteria, population structure, or restart structure.
7. Strategy `S2` is evidence-guided refinement.
8. `S2` asks the LLM to diagnose weaknesses such as poor small-instance gaps, poor large-instance gaps, inconsistent
   results, worst-case brittleness, and timeouts.
9. `S2` asks for one or two targeted changes while preserving useful mechanisms.
10. Strategy `S3` is complementary recombination.
11. `S3` asks the LLM to combine mechanisms from selected parents without concatenating solvers or running them
    sequentially.
12. Every evolution prompt includes selected parent context.
13. Every evolution prompt includes selected parent awareness: scalar parent ranking, invalid/timeout caution,
    archive-specialist cues, and strategy-specific instructions for how to use the parents.
14. Every evolution prompt includes selected parent verbal gradients when enabled.
15. `S1`, `S2`, and `S3` each receive their strategy-specific next-mutation advice from the selected parent gradients.
16. Every evolution prompt includes the same TSP solver contract, internal checklist, and JSON response format used for
    initial prompts.
17. Optional LLM verbal-gradient calls happen only for selected parents missing an LLM gradient and only up to the
    configured per-generation cap.
18. Offspring tasks are executed, evaluated, and persisted through the same candidate task execution path as initial
    tasks.

## 21. Population Update Per Generation

1. After offspring evaluation, the engine registers offspring and updates the archive.
2. The engine concatenates current population candidates and offspring candidates.
3. `select_survivors` selects `population_size` survivors from the combined list.
4. The archive does not force old archive entries back into the active population.
5. A new `Population` object is created for the current generation.
6. `generation_summary` is built from the new survivor population, the just-created offspring list, and compact archive
   counts.
7. `RunStore.save_generation` writes `generation_<NNN>/population.json`, `generation_<NNN>/offspring.json`, and
   `generation_<NNN>/summary.json`.
8. `RunStore.save_archive` writes `archive/archive.json` and `archive/generation_<NNN>.json`.
9. The loop then advances to the next generation.

## 22. Final Search Best Selection

1. After the last configured generation, the engine selects the best search candidate.
2. When `archive.final_selection_uses_archive` is true, final search-best selection considers the final population plus
   archive candidates.
3. The same metric-vector survivor ordering is used, including scalar score for scoreable timeout candidates.
4. The selected candidate is called `search_best`.
5. `search_best` is the only candidate evaluated on the held-out test pool by `EvolutionEngine.run`.

## 23. Offline Test Evaluation

1. `test_evaluator.evaluate_code(search_best.code)` evaluates the best candidate source on the test pool.
2. No LLM calls happen during test evaluation.
3. No mutation or parent selection happens during test evaluation.
4. The test evaluator re-runs static validation on the selected source.
5. The test evaluator executes every `test_instances x seeds` task with the same sandbox, timeout, and budget mechanisms
   as search evaluation.
6. For current TSP configs, test instances are loaded from `data/tsp/test_instances`.
7. Those TSPLIB files include `OPTIMAL` values, so gaps and mean gap are normally available.
8. The final test `EvaluationResult` is independent of the candidate object's search metrics.
9. `RunStore.save_test_result` writes `test_result.json` with candidate ID, status, distance, error details, and test
    metrics.

## 24. Final Reporting

1. `_llm_call_summary` reads the provider summary if available.
2. It ensures `candidate_generation_calls`, `total_api_calls`, and `failed_calls` keys exist.
3. It adds the provider model when available.
4. It recomputes the configured candidate-generation budget.
5. It sets `budget_match` to whether counted candidate-generation calls equal the configured budget.
6. `RunStore.save_llm_calls` writes `llm_calls.json`.
7. `build_final_report` creates `final_report.md`.
8. The final report includes a ranked final population table.
9. The final report includes the search best candidate ID, name, status, search distance, thought, and optional error
    details.
10. The final report includes test status, test distance, instances evaluated, valid/scored counts, partial timeout
    count, mean gap or mean tour length, penalized mean gap, timeout penalty, median gap, worst gap, and best gap.
11. The final report includes candidate-generation calls, reflection/feedback calls, failed calls, verbal-gradient
    counts, archive counts, model, configured candidate-generation budget, and budget match.
12. `EvolutionEngine.run` returns the final `Population`.
13. The CLI prints the run directory path and `best=<candidate_id> distance=<distance>`.

## 25. Output Artifact Map

| Artifact                                   | Writer                        | Contents                                                          |
|--------------------------------------------|-------------------------------|-------------------------------------------------------------------|
| `config.json`                              | `RunStore.create`             | Resolved run configuration                                        |
| `candidates/cand_*.json`                   | `RunStore.save_candidate`     | Candidate metadata, metrics, verbal gradient, prompt text, status, distance, errors |
| `candidates/cand_*.py`                     | `RunStore.save_candidate`     | Generated solver source code                                      |
| `prompts/cand_*_prompt.txt`                | `RunStore.save_candidate`     | Exact prompt sent to the LLM                                      |
| `prompts/cand_*_response.txt`              | `RunStore.save_candidate`     | Raw response text when present                                    |
| `generations/generation_*/population.json` | `RunStore.save_generation`    | Survivor candidates for that generation                           |
| `generations/generation_*/offspring.json`  | `RunStore.save_generation`    | Offspring candidates created in that generation                   |
| `generations/generation_*/summary.json`    | `RunStore.save_generation`    | Best ID, best distance, counts, status counts                     |
| `feedback/generation_*_verbal_gradient_*.json` | `RunStore.save_feedback`  | Optional LLM verbal-gradient prompt, response, status, and parsed gradient |
| `archive/archive.json`                     | `RunStore.save_archive`       | Latest archive state, bucket summaries, and archive entries       |
| `archive/generation_*.json`                | `RunStore.save_archive`       | Per-generation archive snapshot                                  |
| `archive_summary.json`                     | `RunStore.save_archive_summary` | Final archive summary                                           |
| `test_result.json`                         | `RunStore.save_test_result`   | Held-out test evaluation result for search best                   |
| `llm_calls.json`                           | `RunStore.save_llm_calls`     | Counted and configured LLM call information plus compact archive summary |
| `final_report.md`                          | `RunStore.write_final_report` | Human-readable final run report                                   |

## 26. Important TSP-Specific Semantics

1. Search distance and test distance can use different quantities.
2. Search distance for a valid TSP candidate is `mean_tour_length` because the synthetic search pool usually has no
    known optimum.
3. Test distance for a valid TSP candidate is `mean_gap` when TSPLIB optimal lengths are present.
4. Lower distance is always better.
5. A scoreable timeout candidate competes with valid candidates by scalar distance, while timeout fraction and timeout
   status remain robustness penalties inside the metric vector.
6. A partial timeout is only gap-scored when the instance has an optimal reference and the reported tour is valid.
7. `report_best_tour` is useful because a solver that times out can still expose a valid incumbent for partial
   evaluation.
8. The evaluator always validates reported and returned tours; generated code cannot claim validity without producing a
   permutation.
9. The evaluator closes the TSP cycle itself, so candidate solvers must not include the start node twice.
10. The generated solver receives only a distance matrix, seed, and budget. It does not receive coordinates, optimal
    values, file paths, candidate IDs, or run metadata.
11. Candidate tasks and instance evaluations are both parallelized, so saved candidate IDs may be allocated before tasks
    finish, but result lists preserve task order.
12. The run store persists failed and invalid candidates deliberately so later prompts, summaries, and debugging can
    explain what happened.
13. TSP verbal gradients are static by default and optional LLM-augmented for selected parents.
14. A verbal gradient is parent-specific and is inserted only when that candidate is selected as a parent.
15. Generation-wide reflection broadcast is no longer part of the active TSP run path.

## 27. Main Code Path Reference

| Step                             | Main files                                                                                                                                                                      |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CLI and config                   | `dynagen/cli.py`, `dynagen/config.py`                                                                                                                                           |
| Problem selection                | `dynagen/problems/registry.py`, `dynagen/problems/tsp.py`                                                                                                                       |
| TSP data model and parsing       | `dynagen/domain/tsp_instance.py`, `dynagen/domain/tsp_parser.py`, `dynagen/domain/tsp_synthetic.py`, `dynagen/domain/tour.py`                                                   |
| Prompt construction              | `dynagen/prompts/tsp_initial.py`, `dynagen/prompts/tsp_evolution.py`, `dynagen/prompts/tsp_templates.py`                                                                        |
| Evolution loop                   | `dynagen/evolution/engine.py`, `dynagen/evolution/selection.py`, `dynagen/evolution/strategies.py`, `dynagen/evolution/population.py`                                      |
| Candidate parsing and validation | `dynagen/candidates/parser.py`, `dynagen/candidates/validation.py`, `dynagen/candidates/candidate.py`                                                                           |
| LLM providers                    | `dynagen/llm/base.py`, `dynagen/llm/openai_provider.py`, `dynagen/llm/ollama_provider.py`                                                                                       |
| TSP evaluation                   | `dynagen/evaluation/tsp_evaluator.py`, `dynagen/evaluation/tsp_metrics.py`, `dynagen/evaluation/base.py`                                                                        |
| TSP verbal gradients             | `dynagen/evaluation/tsp_gradient.py`, `dynagen/evolution/verbal_gradient.py`                                                                                                    |
| Solver execution                 | `dynagen/execution/tsp_runner.py`, `dynagen/execution/tsp_timeouts.py`, `dynagen/execution/sandbox.py`                                                                          |
| Persistence and reports          | `dynagen/persistence/run_store.py`, `dynagen/persistence/serialization.py`, `dynagen/reporting/summary.py`                                                                      |
