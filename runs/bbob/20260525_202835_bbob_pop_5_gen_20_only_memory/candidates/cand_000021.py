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
        self.local_budget = int(0.1 * budget)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        pop = self.rng.uniform(lb, ub, (n, dim))
        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
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
        while evaluations < self.budget:
            for i in range(n):
                if evaluations >= self.budget:
                    break
                candidates = list(range(n))
                candidates.remove(i)
                a, b, c = self.rng.choice(candidates, 3, replace=False)
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
            # Local search: try a random perturbation around best
            if self.local_budget > 0 and evaluations < self.budget:
                step = 0.1 * (ub - lb)
                x_perturb = best_x + self.rng.normal(0, step)
                x_perturb = np.clip(x_perturb, lb, ub)
                val = func(x_perturb)
                evaluations += 1
                self.local_budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = x_perturb.copy()
                    report_best(best_val, best_x)
        return best_val, best_x