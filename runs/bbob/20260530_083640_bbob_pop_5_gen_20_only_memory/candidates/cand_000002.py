import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        calls = 0

        # Population size: at least 4, at most budget//2, heuristic
        pop_size = max(4, min(200, budget // (dim + 1)))

        # Initial population
        pop = lb + (ub - lb) * self.rng.uniform(size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf

        for i in range(pop_size):
            if calls >= budget:
                break
            val = func(pop[i])
            calls += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # Main DE loop
        while calls < budget:
            for i in range(pop_size):
                if calls >= budget:
                    break
                # Choose three distinct random indices
                candidates = [j for j in range(pop_size) if j != i]
                r = self.rng.choice(candidates, 3, replace=False)
                a, b, c = r[0], r[1], r[2]

                # Mutation: scale factor F
                F = 0.8
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)

                # Crossover probability CR
                CR = 0.9
                cross_points = self.rng.uniform(size=dim) < CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Evaluation
                val = func(trial)
                calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x