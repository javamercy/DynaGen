import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        pop_size = max(4, min(20, budget // 20))
        if pop_size > budget:
            pop_size = budget
        self.pop_size = pop_size
        self.F = 0.8
        self.CR = 0.9
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        pop = lb + (ub - lb) * self.rng.rand(self.pop_size, self.dim)
        pop_fit = np.full(self.pop_size, np.inf)
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = x.copy()
                report_best(self.best_val, self.best_x)
        while evals < self.budget:
            new_pop = pop.copy()
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.randint(self.dim)] = True
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
        return self.best_val, self.best_x