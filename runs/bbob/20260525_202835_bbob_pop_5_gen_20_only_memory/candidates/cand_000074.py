import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(4 * dim, budget // 4))
        self.F = 0.5
        self.CR = 0.5

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop = self.rng.uniform(lb, ub, size=(self.popsize, dim))
        fitness = np.full(self.popsize, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        # Initial evaluations
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
        # DE phase: allocate up to 80% budget
        de_budget = int(0.8 * self.budget)
        while evals < de_budget:
            for i in range(self.popsize):
                if evals >= de_budget:
                    break
                idx_best = np.argmin(fitness)
                r1, r2 = self.rng.choice([j for j in range(self.popsize) if j != i], 2, replace=False)
                mutant = pop[i] + self.F * (pop[idx_best] - pop[i]) + self.F * (pop[r1] - pop[r2])
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
        # Local search phase: random perturbation around best
        remaining = self.budget - evals
        if remaining > 0 and best_x is not None:
            step_size = 0.05 * (ub - lb)  # initial step
            for _ in range(remaining):
                # random direction
                direction = self.rng.normal(size=dim)
                direction /= np.linalg.norm(direction) + 1e-12
                candidate = best_x + step_size * direction
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                step_size *= 0.99  # decrease step
        return best_val, best_x