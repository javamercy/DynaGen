import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = min(max(10, dim * 2), budget // 2)
        pop_size = max(pop_size, 2)
        pop = self.rng.uniform(lb, ub, (pop_size, dim))
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

        mean_point = np.mean(pop, axis=0)
        initial_diversity = np.mean(np.linalg.norm(pop - mean_point, axis=1))
        diversity_threshold = 0.05 * initial_diversity if initial_diversity > 0 else 1e-9

        F = 0.8
        CR = 0.9
        generation = 0

        while evals < budget:
            # Check diversity and restart if necessary
            if generation > 0:
                mean_point = np.mean(pop, axis=0)
                diversity = np.mean(np.linalg.norm(pop - mean_point, axis=1))
                if diversity < diversity_threshold:
                    # Keep best, restart half of the rest
                    best_idx = np.argmin(fitness)
                    restart_indices = [i for i in range(pop_size) if i != best_idx]
                    restart_indices = self.rng.choice(restart_indices, size=pop_size//2, replace=False)
                    for idx in restart_indices:
                        if evals >= budget:
                            break
                        pop[idx] = self.rng.uniform(lb, ub, dim)
                        val = func(pop[idx])
                        evals += 1
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = pop[idx].copy()
                            report_best(best_val, best_x)
                    # Recompute diversity threshold
                    if evals < budget:
                        mean_point = np.mean(pop, axis=0)
                        initial_diversity = np.mean(np.linalg.norm(pop - mean_point, axis=1))
                        diversity_threshold = 0.05 * initial_diversity if initial_diversity > 0 else 1e-9

            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    break
                a, b, c = self.rng.choice(candidates, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[self.rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
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