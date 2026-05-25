import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(8, min(30, budget // 15))
        self.stagnation_limit = max(10, dim)
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        if self.budget == 0:
            return float('inf'), None
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
        generation = 0
        no_improvement = 0
        while evals < self.budget:
            # Mutation and crossover
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                # select 5 distinct random indices
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c, d, e = self.rng.choice(idxs, 5, replace=False)
                # mutation: rand/2
                F = 0.5 + 0.5 * self.rng.rand()
                mutant = pop[a] + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                CR = 0.5 + 0.5 * self.rng.rand()
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
                    no_improvement = 0
                if trial_fit < pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = trial_fit
            # Check for stagnation
            if no_improvement >= self.stagnation_limit:
                # attempt restart of worst half
                half = self.pop_size // 2
                if evals + half <= self.budget:
                    worst_indices = np.argsort(pop_fit)[-half:]
                    for idx in worst_indices:
                        pop[idx] = lb + (ub - lb) * self.rng.rand(self.dim)
                        pop_fit[idx] = func(pop[idx])
                        evals += 1
                        if pop_fit[idx] < self.best_val:
                            self.best_val = pop_fit[idx]
                            self.best_x = pop[idx].copy()
                            report_best(self.best_val, self.best_x)
                    no_improvement = 0
            generation += 1
        return self.best_val, self.best_x