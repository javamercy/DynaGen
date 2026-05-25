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
        self.local_evals_per_gen = max(1, int(0.1 * budget / (budget // self.popsize)))  # limit local search per generation

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        # Initial population
        pop = self.rng.uniform(lb, ub, size=(self.popsize, dim))
        fitness = np.full(self.popsize, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        for i in range(self.popsize):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # Main loop
        while evals < self.budget:
            # DE generation
            for i in range(self.popsize):
                if evals >= self.budget:
                    break
                indices = list(range(self.popsize))
                indices.remove(i)
                a, b, c = self.rng.choice(indices, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.random(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
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
            # Local search on best
            if evals < self.budget:
                best_x, best_val, used = self._local_search(func, best_x, best_val, lb, ub, evals)
                evals += used
        return best_val, best_x

    def _local_search(self, func, x, val, lb, ub, evals):
        dim = self.dim
        step = 0.1 * (ub - lb)
        used = 0
        max_local = min(self.local_evals_per_gen, self.budget - evals)
        if max_local <= 0:
            return x, val, 0
        improvement = True
        while used < max_local:
            if not improvement:
                step *= 0.5
                improvement = True
            improved = False
            perm = self.rng.permutation(dim)
            for i in perm:
                if used >= max_local:
                    break
                # Positive
                x_new = x.copy()
                x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 1.5
                    improved = True
                    continue
                # Negative
                if used >= max_local:
                    break
                x_new = x.copy()
                x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 1.5
                    improved = True
            if not improved:
                improvement = False
        return x, val, used