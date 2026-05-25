import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.popsize = max(4, min(10 * dim, budget // 2))
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = self.rng.uniform(lb, ub, size=(self.popsize, self.dim))
        fitness = np.full(self.popsize, np.inf)
        evaluations = 0
        best_value = np.inf
        best_x = None

        for i in range(self.popsize):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_value:
                best_value = val
                best_x = x.copy()
                report_best(best_value, best_x)

        while evaluations < self.budget:
            for i in range(self.popsize):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.popsize))
                indices.remove(i)
                a, b, c = self.rng.choice(indices, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_value:
                        best_value = val
                        best_x = trial.copy()
                        report_best(best_value, best_x)

        return best_value, best_x