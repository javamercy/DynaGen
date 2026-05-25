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
        if self.budget == 0:
            return float('inf'), None
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # For very small budget, just random search
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
        # Differential evolution
        pop_size = max(4, min(20, self.budget // 20))
        if pop_size > self.budget:
            pop_size = self.budget
        F = 0.8
        CR = 0.9
        pop = lb + (ub - lb) * self.rng.rand(pop_size, self.dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= self.budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        while evals < self.budget:
            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                if len(idxs) < 3:
                    trial = lb + (ub - lb) * self.rng.rand(self.dim)
                    trial_fit = func(trial)
                    evals += 1
                    if trial_fit < self.best_val:
                        self.best_val = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
                    if trial_fit < pop_fit[i]:
                        new_pop[i] = trial
                        pop_fit[i] = trial_fit
                    continue
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
                    pop_fit[i] = trial_fit
            pop = new_pop
        return self.best_val, self.best_x