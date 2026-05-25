import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # population size: small, adapt to dimension and budget
        self.popsize = max(4, min(4 * dim, budget // 2))
        self.F = 0.8
        self.CR = 0.9
        # SA temperature schedule
        self.T0 = 1.0
        self.T_end = 1e-8
        self.alpha = (self.T_end / self.T0) ** (1.0 / budget)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        # initialize population
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # evaluate initial population
        for i in range(popsize):
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
        # main DE loop with SA acceptance
        while evaluations < self.budget:
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                # choose three distinct random indices different from i
                indices = list(range(popsize))
                indices.remove(i)
                a, b, c = self.rng.choice(indices, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                cross_points = self.rng.random(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                # SA acceptance
                delta = val - fitness[i]
                if delta < 0:
                    accept = True
                else:
                    T = self.T0 * (self.alpha ** evaluations)
                    accept = self.rng.random() < np.exp(-delta / T)
                if accept:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
        return best_val, best_x