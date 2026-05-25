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
        evals = 0
        budget = self.budget
        rng = self.rng

        if budget < 3:
            while evals < budget:
                x = lb + (ub - lb) * rng.rand(self.dim)
                val = func(x)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x

        pop_size = min(budget, max(3, min(20, budget // 5)))
        pop = lb + (ub - lb) * rng.rand(pop_size, self.dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        while evals < budget:
            frac = evals / budget
            F = 0.8 * (1 - 0.5 * frac)
            CR = 0.9 * (1 - 0.3 * frac)
            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(self.dim)] = True
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