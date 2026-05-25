import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(10 * dim, budget // 2))
        self.F = 0.8
        self.CR = 0.9
        self.local_budget = int(0.2 * budget)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        # Latin hypercube initialization
        pop = np.empty((n, dim))
        for d in range(dim):
            cut = np.linspace(lb[d], ub[d], n + 1)
            u = np.random.default_rng(self.seed + d).random(n)
            perm = np.random.default_rng(self.seed + 1000 + d).permutation(n)
            pop[perm, d] = cut[:-1] + u * (cut[1:] - cut[:-1])
        pop = np.clip(pop, lb, ub)
        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # Initial evaluation
        for i in range(n):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # Main loop
        while evaluations < self.budget:
            # DE generation using DE/best/1/bin
            for i in range(n):
                if evaluations >= self.budget:
                    break
                indices = list(range(n))
                indices.remove(i)
                a, b = self.rng.choice(indices, 2, replace=False)
                mutant = best_x + self.F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.random(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # Local search on best solution using random perturbations
            if self.local_budget > 0 and evaluations < self.budget:
                best_x, best_val, used = self._local_search(func, best_x, best_val, lb, ub, evaluations)
                evaluations += used
                self.local_budget -= used
        return best_val, best_x

    def _local_search(self, func, x, val, lb, ub, evaluations):
        dim = self.dim
        used = 0
        step_size = 0.1 * (ub - lb)
        while self.local_budget > 0 and evaluations + used < self.budget:
            # Random perturbation
            perturb = self.rng.normal(0, step_size, dim)
            new_x = np.clip(x + perturb, lb, ub)
            new_val = func(new_x)
            used += 1
            self.local_budget -= 1
            if new_val < val:
                val = new_val
                x = new_x.copy()
                report_best(val, x)
                step_size *= 1.5  # adapt step if successful
            else:
                step_size *= 0.9  # shrink step if failed
            # Ensure step_size does not become too small
            step_size = np.maximum(step_size, 1e-10 * (ub - lb))
        return x, val, used