import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10, budget // 20))
        if self.pop_size > budget:
            self.pop_size = budget
        self.F = 0.8
        self.CR = 0.9
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # Initial population
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
        # DE phase: use 60% of budget
        de_budget = int(self.budget * 0.6)
        while evals < de_budget:
            new_pop = pop.copy()
            for i in range(self.pop_size):
                if evals >= de_budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b = self.rng.choice(idxs, 2, replace=False)
                # current-to-best/1 mutation
                mutant = pop[i] + self.F * (self.best_x - pop[i]) + self.F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
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
        # Local search phase: Gaussian perturbation with decaying step
        step_size = 0.1 * (ub - lb)
        while evals < self.budget:
            candidate = self.best_x + step_size * self.rng.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            fit = func(candidate)
            evals += 1
            if fit < self.best_val:
                self.best_val = fit
                self.best_x = candidate.copy()
                report_best(self.best_val, self.best_x)
            # Decay step size
            step_size *= 0.99
            # Prevent step_size from becoming too small
            if np.max(step_size) < 1e-12:
                step_size = 1e-12 * (ub - lb)
        return self.best_val, self.best_x