import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(20, budget // 40))
        self.F = 0.5
        self.CR = 0.9
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None
        self.local_samples = max(1, budget // 200)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # initial population
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
        # main loop
        gen = 0
        while evals < self.budget:
            # DE generation
            new_pop = pop.copy()
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b = self.rng.choice(idxs, 2, replace=False)
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
            # local search around best
            sigma = 0.2 * (ub - lb) * (1 - gen / max(1, self.budget // self.pop_size))
            for _ in range(self.local_samples):
                if evals >= self.budget:
                    break
                perturbation = sigma * self.rng.randn(self.dim)
                trial = np.clip(self.best_x + perturbation, lb, ub)
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
            gen += 1
        return self.best_val, self.best_x