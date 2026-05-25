import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(4 * dim, budget // 2))
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop = self.rng.uniform(lb, ub, size=(self.popsize, dim))
        fitness = np.full(self.popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # evaluate initial population
        for i in range(self.popsize):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # main DE loop
        while evaluations < self.budget:
            for i in range(self.popsize):
                if evaluations >= self.budget:
                    break
                # rand/1 mutation
                candidates = [j for j in range(self.popsize) if j != i]
                r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross_points = self.rng.random(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
        return best_val, best_x