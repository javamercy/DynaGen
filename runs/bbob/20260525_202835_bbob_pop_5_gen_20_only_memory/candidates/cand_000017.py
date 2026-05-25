import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(5 * dim, budget // 3))  # smaller population for more iterations
        self.F = 0.5
        self.CR = 0.5
        self.local_budget = int(0.25 * budget)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        # LHS initialization
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
        # Main DE loop
        while evaluations < self.budget:
            for i in range(self.popsize):
                if evaluations >= self.budget:
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
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # Local search on best if budget remains
            if self.local_budget > 0 and evaluations < self.budget:
                best_x, best_val, used = self._intensive_local_search(func, best_x, best_val, lb, ub, evaluations)
                evaluations += used
                self.local_budget -= used
        return best_val, best_x

    def _intensive_local_search(self, func, x, val, lb, ub, evaluations):
        dim = self.dim
        step = 0.1 * (ub - lb)
        used = 0
        max_iters = 3  # number of full coordinate sweeps
        for _ in range(max_iters):
            if self.local_budget <= 0 or evaluations + used >= self.budget:
                break
            improved_any = False
            # Random order of coordinates
            order = self.rng.permutation(dim)
            for i in order:
                if self.local_budget <= 0 or evaluations + used >= self.budget:
                    break
                # Positive direction
                x_new = x.copy()
                x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                self.local_budget -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 2.0
                    improved_any = True
                    continue
                # Negative direction
                if self.local_budget <= 0 or evaluations + used >= self.budget:
                    break
                x_new = x.copy()
                x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                self.local_budget -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 2.0
                    improved_any = True
                else:
                    step[i] *= 0.5
            if not improved_any:
                # Try random direction perturbation
                if self.local_budget <= 0 or evaluations + used >= self.budget:
                    break
                d = self.rng.normal(0, 1, dim)
                d = d / np.linalg.norm(d)
                scale = np.mean(step) * 0.5
                x_new = np.clip(x + scale * d, lb, ub)
                val_new = func(x_new)
                used += 1
                self.local_budget -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
        return x, val, used