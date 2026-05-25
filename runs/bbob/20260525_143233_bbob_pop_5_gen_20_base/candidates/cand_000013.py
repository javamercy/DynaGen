import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        pop_size = max(4 * dim, 30)
        pop_size = min(pop_size, budget // 2)
        pop_size = max(pop_size, 2)
        self.pop_size = pop_size

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        rng = self.rng

        # Initialize population
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = float('inf')
        best_x = None
        evals = 0

        # Initial evaluation
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Compute initial diversity
        centroid = np.mean(pop, axis=0)
        initial_diversity = np.mean(np.linalg.norm(pop - centroid, axis=1))
        diversity_threshold = 0.05 * initial_diversity if initial_diversity > 0 else 1e-9

        # Main loop
        F_scale = 0.8
        CR = 0.9
        generation = 0

        while evals < budget:
            # Check diversity and restart if needed
            if generation > 0:
                centroid = np.mean(pop, axis=0)
                diversity = np.mean(np.linalg.norm(pop - centroid, axis=1))
                if diversity < diversity_threshold:
                    # Keep best, restart half of the rest
                    best_idx = np.argmin(fitness)
                    restart_indices = [i for i in range(pop_size) if i != best_idx]
                    restart_count = pop_size // 2
                    if restart_count > len(restart_indices):
                        restart_count = len(restart_indices)
                    if restart_count > 0:
                        restart_indices = rng.choice(restart_indices, size=restart_count, replace=False)
                        for idx in restart_indices:
                            if evals >= budget:
                                break
                            pop[idx] = rng.uniform(lb, ub, dim)
                            val = func(pop[idx])
                            evals += 1
                            fitness[idx] = val
                            if val < best_val:
                                best_val = val
                                best_x = pop[idx].copy()
                                report_best(best_val, best_x)
                        # Recompute diversity threshold
                        if evals < budget:
                            centroid = np.mean(pop, axis=0)
                            initial_diversity = np.mean(np.linalg.norm(pop - centroid, axis=1))
                            diversity_threshold = 0.05 * initial_diversity if initial_diversity > 0 else 1e-9

            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation
                if rng.rand() < 0.9:
                    # DE/rand/1 with Cauchy F
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    if len(candidates) < 3:
                        break
                    a, b, c = rng.choice(candidates, size=3, replace=False)
                    F = np.clip(rng.standard_cauchy(), 0, 2)  # Cauchy distributed, clipped
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    # Cauchy perturbation around best
                    scale = (1.0 - evals / budget) * 0.1 * (ub - lb)
                    mutant = best_x + rng.standard_cauchy(dim) * scale
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Evaluate
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            generation += 1

        return best_val, best_x