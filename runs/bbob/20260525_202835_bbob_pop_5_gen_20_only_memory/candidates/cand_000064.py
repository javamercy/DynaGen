import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Determine population size
        popsize = max(4 * dim, 20)
        if popsize > budget // 2:
            popsize = max(2, budget // 2)
        self.popsize = popsize

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        budget = self.budget
        rng = self.rng

        # Initialize population
        pop = rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0

        # Evaluate initial population
        for i in range(popsize):
            if evaluations >= budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Main DE loop
        while evaluations < budget:
            for i in range(popsize):
                if evaluations >= budget:
                    break
                # Mutation: DE/rand/1
                candidates = [j for j in range(popsize) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = pop[r1] + 0.5 * (pop[r2] - pop[r3])
                # Clip to bounds
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                cross_points = rng.random(dim) < 0.9
                if not np.any(cross_points):
                    cross_points[rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # Evaluate
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