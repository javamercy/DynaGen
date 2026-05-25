import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.popsize = max(4, min(10 * dim, budget // 2))
        self.F_start = 0.9
        self.F_end = 0.4
        self.CR_start = 0.9
        self.CR_end = 0.6

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

        max_gen = (self.budget - evaluations) // self.popsize
        generation = 0
        while evaluations < self.budget:
            if max_gen > 0:
                t = generation / max_gen
            else:
                t = 0
            F = self.F_start + (self.F_end - self.F_start) * t
            CR = self.CR_start + (self.CR_end - self.CR_start) * t

            for i in range(self.popsize):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.popsize))
                indices.remove(i)
                a, b = self.rng.choice(indices, 2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.rand(self.dim) < CR
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
            generation += 1

        return best_value, best_x