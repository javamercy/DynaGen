# Archive-Conditioned Evolution Implementation Guide

This guide specifies the archive-conditioned evolution feature for DynaGen. The current implementation follows this
design at the generic engine level, with problem-specific archive profiles for TSP, BBOB, and DVRP.

Archive-conditioned evolution adds a persistent quality-diversity memory beside the current population. The current
population stays small and generation-local. The archive stores strong, robust, and specialist candidates discovered at
any point in the run, then conditions future parent selection, recombination, prompting, reporting, and final
search-best
selection.

## Intended Bias

The feature should deliberately bias DynaGen toward:

1. **Retaining useful specialists.** A candidate that is excellent on one instance family should not disappear only
   because its scalar mean score is weaker than a generalist in the current generation.
2. **Reusing old discoveries.** Early-generation candidates can contain mechanisms that later mutations lose. The
   archive
   should keep those mechanisms available as parents.
3. **Improving recombination quality.** S3 should combine complementary parents, not just three rank-nearby survivors.
4. **Protecting robustness.** Worst-group and timeout behavior should influence archive membership, not only mean score.
5. **Keeping exploitation controlled.** The archive should increase the chance of reusing elites without collapsing
   every
   prompt into the same parent family.
6. **Staying problem-aware.** The engine should expose generic archive hooks, while each problem defines its own niche
   labels and archive score vectors.

This is a quality-diversity bias, not just an elitism cache. A pure top-k archive would mostly duplicate the current
survivor set. The intended behavior is to keep both global elites and niche elites.

## Current Behavior To Replace Or Extend

Current evolution uses only the active population for parent selection:

1. Generation 0 candidates are evaluated.
2. `select_survivors` keeps `population_size` candidates by status, scalar score, runtime, generation, and ID.
3. Each offspring task selects parents from the current population with rank-biased probabilities.
4. Offspring and current survivors are merged.
5. `select_survivors` creates the next population.
6. The final search-best candidate is selected from the final population only.

The archive should extend this path without changing the candidate-generation budget. Archive operations are
deterministic
bookkeeping over already-evaluated candidates.

## Generic Specification

### Config

Add an `ArchiveConfig` under `EvolutionConfig`.

```yaml
evolution:
  archive:
    enabled: true
    max_size: 64
    max_per_bucket: 4
    add_statuses: [ "valid", "evaluated", "timeout" ]
    parent_sample_probability: 0.35
    s3_archive_parent_min: 1
    final_selection_uses_archive: true
    deduplicate_code: true
    diversity_weight: 0.25
    recency_weight: 0.05
    robustness_weight: 0.35
```

Recommended defaults:

| Field                          |                             Default | Purpose                                                                                        |
|--------------------------------|------------------------------------:|------------------------------------------------------------------------------------------------|
| `enabled`                      |                              `true` | Enables archive update, parent sampling, persistence, and reporting.                           |
| `max_size`                     |                                `64` | Hard cap across all archive entries.                                                           | 
| `max_per_bucket`               |                                 `4` | Keeps each niche from being dominated by near-duplicates.                                      |
| `add_statuses`                 | `["valid", "evaluated", "timeout"]` | Allows scoreable timeout candidates with reported incumbents, but excludes invalid/error code. |
| `parent_sample_probability`    |                              `0.35` | Probability that a parent slot is sampled from the archive instead of the current population.  |
| `s3_archive_parent_min`        |                                 `1` | Ensures recombination usually receives at least one archived parent.                           |
| `final_selection_uses_archive` |                              `true` | Selects the final search candidate from population plus archive.                               |
| `deduplicate_code`             |                              `true` | Prevents repeated code bodies from occupying multiple archive slots.                           |
| `diversity_weight`             |                              `0.25` | Rewards structural or behavioral novelty within a bucket.                                      |
| `recency_weight`               |                              `0.05` | Prevents the archive from becoming permanently frozen.                                         |
| `robustness_weight`            |                              `0.35` | Rewards worst-group and timeout-resistant behavior.                                            |

These defaults intentionally keep the current population influential. The archive should supply additional memory and
complementary parents, not fully replace generation-local selection.

### Data Model

Add a generic archive entry representation. It should store metadata only; candidate code should continue to live in the
normal candidate artifact files.

```python
@dataclass
class ArchiveEntry:
    candidate_id: str
    problem: str
    generation: int
    status: str
    score_name: str
    score_value: float | None
    buckets: list[str]
    primary_bucket: str
    archive_score: float
    quality_score: float
    robustness_score: float
    diversity_score: float
    recency_score: float
    code_hash: str | None
    metrics_snapshot: dict[str, Any]
    bucket_scores: dict[str, float]
    diversity_features: dict[str, Any]
    created_at: str
```

Store archive state separately from the current population:

```text
runs/<problem>/<run>/archive/archive.json
runs/<problem>/<run>/archive/generation_000.json
runs/<problem>/<run>/archive/generation_001.json
```

The generation snapshots make archive dynamics inspectable for papers and ablations.

### Problem Interface

Extend the problem protocol with archive hooks:

```python
def build_archive_profile(self, candidate: Candidate) -> dict[str, Any]:
    ...
```

The profile should return:

```python
{
    "buckets": ["global", "..."],
    "primary_bucket": "...",
    "quality_score": 0.0,
    "robustness_score": 0.0,
    "bucket_scores": {"global": 0.0},
    "diversity_features": {...},
    "metrics_snapshot": {...},
}
```

The engine should remain generic. Problem classes decide what "niche", "robust", and "diverse" mean.

### Archive Update

Archive update should happen after every candidate evaluation and after survivor selection.

Recommended flow:

1. Evaluate candidate.
2. Attach static verbal gradient.
3. Register candidate in `_candidate_index`.
4. If archive is enabled, build the problem-specific archive profile.
5. Reject candidates with disallowed statuses.
6. Reject non-finite score candidates unless the problem profile explicitly marks them as scoreable timeout candidates.
7. Deduplicate by normalized code hash when enabled.
8. Insert the candidate into every listed bucket.
9. Enforce `max_per_bucket`.
10. Enforce global `max_size`.
11. Persist archive state and generation snapshot.

Bucket pruning should be deterministic. Sort entries by:

```text
archive_score desc, quality_score desc, robustness_score desc, generation desc, candidate_id asc
```

For distance-minimization problems, convert lower distance/gap into a higher normalized quality score inside the
problem-specific profile. For BBOB, use AOCC-style higher-is-better quality internally.

### Parent Selection

Parent selection should use a mixed source:

1. Build the usual current-population pool.
2. Build an archive parent pool from eligible archived candidates.
3. For each parent slot, sample from archive with `parent_sample_probability`; otherwise sample from current population.
4. Never select the same candidate twice for one offspring task.
5. If the sampled source is empty, fall back to the other source.
6. For S3, ensure at least `s3_archive_parent_min` archived parents when enough eligible archived entries exist.
7. Prefer complementary archive entries for S3 by sampling different primary buckets where possible.

The prompt does not need a new archive-specific section at first. Archived parents are still normal `Candidate` objects,
so existing parent context and per-candidate verbal gradients should work. Add only a compact marker in the rendered
parent context:

```text
Archive source: yes
Archive bucket: tsp:size:100
Archive role: size specialist
```

This keeps the feature visible to the LLM without bloating prompts.

### Survivor And Final Selection

The active population should still be selected from current population plus offspring. Do not automatically inject
archive
entries into every generation's survivor set; that would reduce exploration and make the population stale.

Use the archive in two additional places:

1. **Parent source:** archived candidates can be selected as parents even when absent from the active population.
2. **Final search-best:** when `final_selection_uses_archive` is true, select the final search-best candidate from
   `final_population + archive_candidates`.

Final selection from the archive should use the same selector as normal survivors unless a later validation tournament
is
implemented. The candidate must already have a search evaluation from the same run.

### Archive Scoring

Use a generic weighted score:

```text
archive_score =
    quality_score
  + robustness_weight * robustness_score
  + diversity_weight * diversity_score
  + recency_weight * recency_score
```

The problem profile should normalize each component to roughly `[0, 1]`.

Quality should represent the main objective. Robustness should represent worst-group behavior, timeout resistance, and
validity. Diversity should represent a difference in behavior or structure from candidates already in the same bucket.
Recency should be a small tie-breaker, not a dominant pressure.

### Reporting

Add archive reporting to `generation_summary`, `llm_calls.json` or a new `archive_summary.json`, and `final_report.md`.

Minimum report fields:

```json
{
  "enabled": true,
  "size": 32,
  "max_size": 64,
  "bucket_count": 12,
  "added_count": 8,
  "rejected_duplicate_count": 3,
  "parent_selections_from_archive": 11,
  "final_selection_from_archive": true,
  "top_buckets": [
    {
      "bucket": "global",
      "candidate_id": "cand_000018",
      "archive_score": 0.91
    }
  ]
}
```

For paper analysis, also report:

1. Percentage of offspring with at least one archived parent.
2. Best archived candidate by generation.
3. Archive bucket occupancy over time.
4. Whether the final tested candidate came from the active population or archive.

### Tests

Add focused unit tests before broad smoke tests:

1. Config parsing and validation for `evolution.archive`.
2. Archive accepts valid/evaluated/scoreable-timeout candidates and rejects invalid/error candidates.
3. Code-hash deduplication keeps the stronger candidate.
4. Per-bucket and global caps are deterministic.
5. Parent selection respects no duplicate parents in one task.
6. S3 receives at least the configured number of archive parents when available.
7. Archive candidates still receive verbal-gradient prompt context.
8. Final search-best can come from archive when enabled.
9. Archive disabled preserves current behavior.

## TSP Guide

### TSP Intended Bias

For TSP, the archive should bias evolution toward retaining solvers that are strong on different instance regimes:

1. Small synthetic instances where more complete local search is affordable.
2. Larger instances where cheap construction and bounded local improvement matter.
3. TSPLIB-style structured instances where geometry-aware heuristics can help.
4. Timeout-resistant candidates that report incumbents early and scale loops by budget.
5. Candidates with complementary construction and improvement mechanisms.

The main goal is to avoid losing a useful size/source specialist when scalar mean tour length favors another candidate.

### TSP Archive Profile

Use existing TSP metrics:

| Signal                | Metric key                                                    |
|-----------------------|---------------------------------------------------------------|
| Main quality          | `distance` or `mean_tour_length` on search pool               |
| Gap robustness        | `mean_gap`, `median_gap`, `worst_gap`, `best_gap`             |
| Runtime               | `mean_runtime`                                                |
| Timeout behavior      | `timeout_fraction`, `timeout_count`, `unscored_timeout_count` |
| Validity              | `valid_count`, `invalid_tour_count`, `runtime_error_count`    |
| Size specialization   | `score_by_instance_size`                                      |
| Source specialization | `score_by_instance_source`                                    |

Recommended buckets:

```text
global
tsp:size:<dimension>
tsp:source:<source>
tsp:runtime:fast
tsp:runtime:robust
tsp:gap:worst_case
tsp:mechanism:<feature>
```

Initial mechanism buckets can be heuristic static code features:

| Bucket                           | Detection hint                                                   |
|----------------------------------|------------------------------------------------------------------|
| `tsp:mechanism:nearest_neighbor` | code contains nearest-neighbor style min-distance selection      |
| `tsp:mechanism:two_opt`          | code contains 2-opt, segment reversal, or edge-swap loops        |
| `tsp:mechanism:insertion`        | code contains insertion or cheapest insertion logic              |
| `tsp:mechanism:random_restart`   | code uses repeated seeds, shuffling, or restart loops            |
| `tsp:mechanism:candidate_list`   | code limits neighborhoods to nearest candidates or bounded lists |

Static feature detection should be conservative. It is better to miss a mechanism than to assign many false labels.

### TSP Quality And Robustness Scores

Use lower-is-better metrics and convert them to higher-is-better archive scores.

Recommended quality:

```text
quality_score = normalized_inverse(mean_tour_length or distance)
```

Recommended robustness:

```text
robustness_score =
    0.45 * normalized_inverse(worst_gap)
  + 0.30 * (1 - timeout_fraction)
  + 0.15 * valid_count / runs
  + 0.10 * normalized_inverse(mean_runtime)
```

If the search pool lacks optimal lengths and gaps are unavailable, use tour length, timeout fraction, validity, and
runtime
only.

### TSP Parent Sampling Bias

For S1:

Prefer one archive parent from a different mechanism bucket than recent population elites. The prompt should encourage
exploring a new algorithmic path while preserving the archived parent’s useful scaling or reporting behavior.

For S2:

Prefer the current-population parent unless the archive contains a strictly better candidate in the same primary bucket.
This keeps refinement local and avoids turning S2 into broad recombination.

For S3:

Prefer complementary buckets:

```text
one global or size elite
one mechanism-diverse archive specialist
one current survivor with recent improvement
```

Avoid selecting three candidates from the same `tsp:mechanism:*` bucket when alternatives exist.

### TSP Expected Benefits

TSP should benefit because candidate performance is often instance-regime dependent. A solver with expensive 2-opt may
be
excellent on small instances and bad on larger ones. A cheap constructive solver may scale well but lack precision. The
archive lets DynaGen recombine these mechanisms later instead of depending on both candidates surviving the same
generation.

Specific expected gains:

1. Better large-instance generalization through retained size specialists.
2. Lower timeout rates by preserving fast incumbent-reporting candidates.
3. Better S3 recombination through explicit mechanism diversity.
4. Less run-to-run volatility because early strong candidates remain available.
5. More interpretable paper artifacts through bucket occupancy and archive-parent lineage.

## BBOB Guide

### BBOB Intended Bias

For BBOB, the archive should preserve optimizer candidates that specialize by function group and failure mode:

1. Separable functions.
2. Moderate and ill-conditioned functions.
3. Multimodal functions.
4. Weak-structure or noisy-looking landscapes.
5. Strong final-error refinement candidates.
6. Broad AOCC generalists.

Scalar mean fitness can hide this specialization. A candidate that is weak globally may contain a useful restart policy,
coordinate adaptation rule, or local refinement mechanism for one BBOB group.

### BBOB Archive Profile

Use existing BBOB metrics:

| Signal                  | Metric key                                                    |
|-------------------------|---------------------------------------------------------------|
| Main quality            | `fitness`, `mean_aocc`, `penalized_mean_aocc`                 |
| Final convergence       | `mean_final_error`, `best_final_error`, `worst_final_error`   |
| Function specialization | `aocc_by_function`, `final_error_by_function`                 |
| Group specialization    | `aocc_by_group`                                               |
| Runtime/evaluations     | `mean_runtime`, `mean_evaluations`                            |
| Timeout behavior        | `timeout_fraction`, `timeout_count`, `unscored_timeout_count` |
| Validity                | `valid_count`, `invalid_count`, `runtime_error_count`         |

Recommended buckets:

```text
global
bbob:group:<group>
bbob:function:<function_id>
bbob:final_error:strong
bbob:runtime:fast
bbob:timeout:robust
bbob:mechanism:<feature>
```

Mechanism buckets can start with static features:

| Bucket                                  | Detection hint                                       |
|-----------------------------------------|------------------------------------------------------|
| `bbob:mechanism:random_search`          | uniform sampling without stateful adaptation         |
| `bbob:mechanism:evolution_strategy`     | population, mutation strength, elite selection       |
| `bbob:mechanism:differential_evolution` | difference vectors, crossover, `F`, `CR`             |
| `bbob:mechanism:cma_like`               | covariance, mean vector, step-size adaptation        |
| `bbob:mechanism:restart`                | restarts, stagnation detection, repeated populations |
| `bbob:mechanism:local_refine`           | coordinate descent, pattern search, hill climb       |

### BBOB Quality And Robustness Scores

BBOB already has higher-is-better AOCC. Use that directly when available.

Recommended quality:

```text
quality_score = penalized_mean_aocc or mean_aocc
```

Recommended robustness:

```text
robustness_score =
    0.40 * worst_group_aocc
  + 0.25 * (1 - timeout_fraction)
  + 0.20 * normalized_inverse(mean_final_error)
  + 0.15 * valid_count / runs
```

For bucket-specific ranking, use the bucket’s local AOCC value first, then global quality. For example,
`bbob:group:multimodal` should rank by `aocc_by_group["multimodal"]` before `mean_aocc`.

### BBOB Parent Sampling Bias

For S1:

Prefer archive parents from underrepresented mechanism buckets or weak current groups. If the current population is
mostly
population-based search, sample a local-refinement or restart specialist.

For S2:

Prefer a same-group archive elite when the selected parent has a clear weak group. The verbal gradient can identify the
weak group; the archive supplies a parent that previously handled it better.

For S3:

Prefer one global AOCC elite, one group specialist, and one mechanism-diverse parent. This is useful because BBOB
optimizers often need both broad search and local exploitation.

### BBOB Expected Benefits

BBOB should benefit because different optimizer mechanisms solve different landscape classes. Archive conditioning keeps
useful mechanisms from disappearing when their average score is temporarily worse.

Specific expected gains:

1. Better group-level AOCC by retaining specialists.
2. Better final-error behavior by preserving local-refinement candidates.
3. More robust recombination of exploration, adaptation, and restart mechanisms.
4. Lower timeout or invalid rates by keeping robust bounded-budget candidates.
5. More useful analysis tables: archive occupancy by BBOB group can support paper claims about specialization.

## DVRP Guide

### DVRP Intended Bias

For DVRP, the archive should preserve dispatch heuristics that specialize by fleet, instance size, waiting behavior, and
completion reliability:

1. Small instances where more detailed scoring is affordable.
2. Large instances where dispatch scoring must be cheap.
3. Different truck-count regimes.
4. Heuristics that reduce unnecessary waiting.
5. Heuristics that complete more customers under dynamic reveal constraints.
6. Timeout-resistant candidates that avoid expensive per-decision global scans.

DVRP is especially vulnerable to losing useful dispatch rules because a heuristic can look mediocre globally while being
excellent under one truck-count or instance-size regime.

### DVRP Archive Profile

Use existing DVRP metrics:

| Signal                | Metric key                                                    |
|-----------------------|---------------------------------------------------------------|
| Main quality          | `distance`, `mean_gap`, `penalized_mean_gap`, `mean_makespan` |
| Robustness            | `worst_gap`, `median_gap`, `best_gap`                         |
| Dispatch behavior     | `mean_decisions`, `mean_waits`, `mean_completed_count`        |
| Size specialization   | `score_by_instance_size`                                      |
| Fleet specialization  | `score_by_truck_count`                                        |
| Source specialization | `score_by_instance_source`                                    |
| Runtime               | `mean_runtime`                                                |
| Timeout behavior      | `timeout_fraction`, `timeout_count`                           |
| Validity              | `valid_count`, `invalid_count`, `runtime_error_count`         |

Recommended buckets:

```text
global
dvrp:size:<dimension>
dvrp:trucks:<truck_count>
dvrp:source:<source>
dvrp:waits:low
dvrp:completion:high
dvrp:runtime:fast
dvrp:mechanism:<feature>
```

Mechanism buckets can start with static features:

| Bucket                             | Detection hint                                             |
|------------------------------------|------------------------------------------------------------|
| `dvrp:mechanism:nearest_available` | prioritizes nearest visible customer                       |
| `dvrp:mechanism:urgency`           | uses due time, reveal time, slack, or deadline pressure    |
| `dvrp:mechanism:fleet_balance`     | references truck load, idle trucks, route balance          |
| `dvrp:mechanism:wait_control`      | explicitly scores waiting versus moving                    |
| `dvrp:mechanism:lookahead`         | scores future/revealed customers or projected availability |
| `dvrp:mechanism:return_policy`     | explicitly decides depot return behavior                   |

### DVRP Quality And Robustness Scores

Use lower-is-better gap or makespan. Include completion and waiting behavior because they explain dispatch quality.

Recommended quality:

```text
quality_score = normalized_inverse(penalized_mean_gap or mean_gap or mean_makespan)
```

Recommended robustness:

```text
robustness_score =
    0.30 * normalized_inverse(worst_gap)
  + 0.20 * (1 - timeout_fraction)
  + 0.20 * valid_count / runs
  + 0.15 * normalized_completion
  + 0.15 * normalized_inverse(mean_waits)
```

If gap is unavailable for some instances, use makespan and completion count. Do not over-reward low waits if completion
falls; low waits are useful only when completion remains strong.

### DVRP Parent Sampling Bias

For S1:

Prefer archive parents from a different dispatch mechanism bucket, especially if the active population has converged on
simple nearest-customer rules.

For S2:

Prefer same-regime archive specialists. If a parent is weak on large instances, sample from `dvrp:size:<large>` or
`dvrp:runtime:fast`. If it has high waits, sample from `dvrp:waits:low`.

For S3:

Prefer one quality elite, one regime specialist, and one behavioral specialist:

```text
global or low-gap candidate
size/truck-count specialist
low-wait or high-completion candidate
```

Avoid combining multiple expensive lookahead candidates unless at least one parent has strong runtime robustness.

### DVRP Expected Benefits

DVRP should benefit because dispatch policies have clear regime-specific tradeoffs. The archive can keep a cheap
scalable
heuristic, a low-wait heuristic, and a high-completion heuristic available for later recombination even when only one is
in the active population.

Specific expected gains:

1. Better large-instance scalability by retaining fast dispatch specialists.
2. Lower wait counts without sacrificing completion.
3. Better truck-count generalization through fleet-regime buckets.
4. More stable recombination because S3 can combine quality, scalability, and behavior specialists.
5. Better paper diagnostics through archive lineage and bucket-level performance.

## Implementation Order

1. Add `ArchiveConfig` parsing and validation.
2. Add archive entry, archive state, scoring helpers, and deterministic pruning.
3. Add `RunStore` persistence for archive state and generation snapshots.
4. Add `Problem.build_archive_profile(candidate)` to the protocol.
5. Implement TSP archive profiles and tests.
6. Implement BBOB archive profiles and tests.
7. Implement DVRP archive profiles and tests.
8. Update `EvolutionEngine` to update the archive after evaluation and register archive candidates.
9. Update parent selection to sample from current population plus archive according to config.
10. Add archive source annotations to candidate rendering.
11. Use archive candidates in final search-best selection when enabled.
12. Add reporting fields and docs flow updates.
13. Run focused unit tests, then one tiny smoke run per problem.

## Risks And Feedback

Risk: The archive could reduce exploration by repeatedly selecting old elites.

Decision: Keep `parent_sample_probability` below 0.5 by default, keep active population selection unchanged, and use
recency only as a small tie-breaker.

Risk: A top-k archive would duplicate the current population.

Decision: Use bucketed quality-diversity with `max_per_bucket`.

Risk: Problem-specific logic could leak into the engine.

Decision: The engine manages archive lifecycle and sampling only. Problems build archive profiles.

Risk: Prompt bloat could increase if archive metadata is verbose.

Decision: Archived parents should be rendered like normal parents with one compact archive marker. Existing verbal
gradients remain the main feedback text.

Risk: Archive candidates might come from stale evaluation conditions.

Decision: Archive only candidates evaluated in the current run and current search pool. Cross-run archives should be a
separate future feature.

Risk: Final selection from archive may overfit the search pool.

Decision: Allow final archive selection, but report whether the final candidate came from the archive. Later validation
tournament work should evaluate population plus archive elites on a held-out validation pool.
