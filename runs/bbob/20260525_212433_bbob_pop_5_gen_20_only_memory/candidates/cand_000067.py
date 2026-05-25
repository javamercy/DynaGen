import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(20, budget // 20))
        if self.pop_size > budget:
            self.pop_size = budget
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        if self.budget <= 0:
            return float('inf'), None
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # Initial random search for very small budgets
        if self.budget < 4:
            for _ in range(self.budget):
                x = lb + (ub - lb) * self.rng.rand(self.dim)
                val = func(x)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x
        # Initialize population
        pop = lb + (ub - lb) * self.rng.rand(self.pop_size, self.dim)
        pop_fit = np.full(self.pop_size, np.inf)
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # Main DE loop
        while evals < self.budget:
            fraction = evals / max(1, self.budget)
            F = 0.9 - fraction * 0.7  # from 0.9 to 0.2
            CR = 0.9 - fraction * 0.7  # from 0.9 to 0.2
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b = self.rng.choice(idxs, 2, replace=False)
                # current-to-best/1 mutation
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross_points = self.rng.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[self.rng.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if trial_fit < pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = trial_fit
        return self.best_val, self.best_x