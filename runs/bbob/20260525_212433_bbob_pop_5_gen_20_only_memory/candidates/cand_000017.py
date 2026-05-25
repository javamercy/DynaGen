import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10, budget // 40))
        if self.pop_size > budget:
            self.pop_size = budget
        if self.pop_size < 3:
            self.pop_size = 3
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
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        de_budget = int(0.8 * self.budget)
        while evals < min(self.budget, de_budget):
            fraction = evals / max(1, de_budget)
            F = 0.9 - fraction * 0.7
            CR = 0.9 - fraction * 0.4
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            for i in range(self.pop_size):
                if evals >= min(self.budget, de_budget):
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
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
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
            pop = new_pop
            pop_fit = new_fit
        while evals < self.budget:
            fraction = evals / self.budget
            sigma = 0.1 * (1 - fraction) * (ub - lb).mean()
            candidate = self.best_x + sigma * self.rng.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < self.best_val:
                self.best_val = val
                self.best_x = candidate.copy()
                report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x