import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(4, min(20, budget // 20))
        self.F = 0.5
        self.CR = 0.9
        self.rng = np.random.RandomState(seed)
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
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        while evals < self.budget:
            new_pop = pop.copy()
            # find current best in population
            best_idx = np.argmin(pop_fit)
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                # select three distinct indices different from i and best_idx (but best_idx can be used as base)
                choices = [j for j in range(self.pop_size) if j != i and j != best_idx]
                if len(choices) < 2:
                    # fallback: if not enough distinct, use all except i
                    choices = [j for j in range(self.pop_size) if j != i]
                a, b = self.rng.choice(choices, 2, replace=False)
                mutant = pop[best_idx] + self.F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross_points = self.rng.rand(self.dim) < self.CR
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