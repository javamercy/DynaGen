import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Population size: at least 5, at most 10*dim, but limited by budget/2 to allow evolution
        self.pop_size = max(5, min(10 * dim, budget // 2))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = self.rng.uniform(lb, ub, size=(self.pop_size, self.dim))
        fit = np.full(self.pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Initial evaluation
        for i in range(self.pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            fit[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(val, x)
            if evals >= self.budget:
                break

        # Main DE loop
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                # Mutation: select three distinct random indices different from i
                indices = [j for j in range(self.pop_size) if j != i]
                if len(indices) < 3:
                    break
                r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
                # Rand/1 mutation
                F = 0.8
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                CR = 0.9
                cross_points = self.rng.random(self.dim) < CR
                # Ensure at least one dimension from mutant
                cross_points[self.rng.integers(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fit[i]:
                    pop[i] = trial
                    fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(val, trial)

        return best_val, best_x