# Framework Flow

This is the current DynaGen TSP run path. For the full step-by-step description, see
[`docs/tsp_flow.md`](tsp_flow.md).

```mermaid
flowchart TD
    A[CLI run command<br/>python3 -m dynagen.cli run --config configs/tsp/*.yaml] --> B[Load RunConfig<br/>run, llm, evolution, evaluation, data<br/>problem.type defaults to tsp]
    B --> C[Create CountingLLMProvider<br/>OpenAI or Ollama<br/>configured calls = population_size + generations * strategies * offspring_per_strategy]
    B --> D[TSPProblem from problem_for_config]
    D --> E[Build search evaluator<br/>pool_name = search_instances]
    D --> F[Build test evaluator<br/>pool_name = test_instances]
    E --> G[Load search TSP instances<br/>single or grid synthetic LLaMEA spec, or TSPLIB file/dir]
    F --> H[Load test TSP instances<br/>TSPLIB .tsp files]
    G --> I[Search evaluation tasks<br/>search_instances x seeds]
    H --> J[Test evaluation tasks<br/>test_instances x seeds]
    B --> K[Create RunStore<br/>runs/tsp/timestamp_name<br/>config.json, candidates, prompts, generations, feedback, archive]
    C --> L[EvolutionEngine.run]
    I --> L
    J --> L
    K --> L
    L --> M[Generation 0 initial population<br/>create TSP roles and prompts]
    M --> N[Parallel candidate tasks<br/>max 8 workers]
    N --> O[LLM complete<br/>JSON schema response: name, thought, code]
    O --> P{Response parsed?}
    P -- No --> Q[Create error candidate<br/>distance = inf<br/>empty metrics]
    P -- Yes --> R[Create Candidate<br/>status = pending]
    R --> S[TSPCandidateEvaluator.evaluate_candidate]
    S --> T[Static code validation<br/>AST, allowed imports, solve_tsp signature]
    T --> U{Valid code?}
    U -- No --> V[Return invalid evaluation<br/>distance = inf<br/>empty metrics]
    U -- Yes --> W[Run every instance x seed<br/>ThreadPool max 8]
    W --> X[run_tsp_solver]
    X --> Y[execute_tsp_solver_code<br/>separate process, timeout_seconds]
    Y --> Z[Sandbox generated code<br/>solve_tsp — distance_matrix.copy, seed, budget<br/>report_best_tour captures incumbent]

Z --> AA{Solver result}
AA -- ok --> AB[Validate returned tour<br/>permutation of node ids]
AA -- timeout with reported tour --> AC[Validate reported best tour<br/>partial = true]
AA -- timeout without report --> AD[Timeout without scoreable tour]
AA -- process error --> AE[Runtime error]

AB --> AF[Compute tour length<br/>cycle is closed by evaluator]
AC --> AF
AF --> AG[Compute gap if optimal_length exists<br/>100 * (tour_length - optimal) / optimal]
AD --> AH[Record timeout]
AE --> AI[Record error]
V --> AJ[No solver runs are launched]

AG --> AK[Aggregate TSP metrics<br/>mean_tour_length, mean_gap, timeout counts, score groups, records]
AH --> AK
AI --> AK
AJ --> AK
AK --> AL[Set candidate status and distance<br/>search: mean_tour_length when valid<br/>test: mean_gap when available<br/>timeout: timeout_distance<br/>invalid/error: inf]
Q --> AL2[Attach static verbal gradient<br/>metrics.verbal_gradient]
AL --> AL2[Attach static verbal gradient<br/>metrics.verbal_gradient]
    AL2 --> AM[Persist candidate artifacts<br/>candidates/*.json, candidates/*.py, prompts/*.txt]
    AM --> AM2{Task batch source?}

    AM2 -- initial --> AN0[Update archive<br/>quality-diversity buckets, code dedupe, archive snapshot]
    AN0 --> AN[Select generation 0 survivors<br/>status rank, distance, id]
    AN --> AO[Save generation_000<br/>population.json, offspring.json, summary.json, archive]
AO --> AP{More generations?}

AP -- Yes --> AQ[Build offspring tasks<br/>for each strategy S1, S2, S3 and offspring_per_strategy]
    AQ --> AR[Select parents<br/>current population mixed with archive<br/>S1/S2: 1 parent, S3: 3 parents]
AR --> AS[Ensure selected parent gradients<br/>static cached on every candidate<br/>optional capped LLM gradient per parent]
AS --> AT[Build TSP evolution prompt<br/>strategy instructions, selected parent verbal gradients, parent context, solver contract]
AT --> N

    AM2 -- offspring --> AU0[Update archive<br/>current-run evaluated candidates only]
    AU0 --> AU[Combine current population and offspring]
    AU --> AV[Select survivors<br/>population_size by status, distance, id<br/>archive does not force survivor re-entry]
AV --> AW[Save generation_N artifacts]
AW --> AP

    AP -- No --> AX[Choose best search candidate<br/>final population plus archive when enabled]
AX --> AY[Offline test evaluation<br/>best code x test_instances x seeds<br/>no LLM, no mutation]
AY --> AZ[Save test_result.json]
AZ --> BA[Save llm_calls.json]
BA --> BB[Write final_report.md]
```
