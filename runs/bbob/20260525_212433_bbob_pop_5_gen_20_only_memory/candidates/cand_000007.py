import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        evals = 0
        if budget < 2:
            x0 = lb + (ub - lb) * rng.rand(dim)
            val = func(x0)
            evals += 1
            if val < self.best_val:
                self.best_val = val
                self.best_x = x0.copy()
                report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x

        pop_size = max(4, min(10, budget // 20))
        de_budget = int(budget * 0.7)
        if de_budget < pop_size:
            de_budget = pop_size

        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        F = 0.5
        CR = 0.9
        while evals < de_budget:
            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= de_budget:
                    break
                best_idx = np.argmin(pop_fit)
                idxs = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                mutant = pop[best_idx] + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if val < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = val
            pop = new_pop

        scale = np.maximum(1e-8, (ub - lb) * 0.01)
        while evals < budget:
            noise = rng.randn(dim) * scale
            trial = self.best_x + noise
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evals += 1
            if val < self.best_val:
                self.best_val = val
                self.best_x = trial.copy()
                report_best(self.best_val, self.best_x)

        return self.best_val, self.best_x