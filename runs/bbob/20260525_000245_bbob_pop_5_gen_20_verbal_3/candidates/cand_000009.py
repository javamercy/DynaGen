import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.default_rng(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        # Population size heuristic
        npop = max(4, min(20, budget // (dim * 5)))
        # Initialize population
        pop = rng.uniform(lb, ub, size=(npop, dim))
        pop_fitness = np.full(npop, np.inf)
        best_x = None
        best_val = np.inf
        calls = 0
        for i in range(npop):
            if calls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            calls += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # If budget exhausted, return
        if calls >= budget:
            return best_val, best_x
        sigma0 = 0.2 * np.mean(ub - lb)
        # Main loop
        while calls < budget:
            # Generate offspring
            if rng.uniform() < 0.2:
                offspring = rng.uniform(lb, ub)
            else:
                remaining = budget - calls
                sigma = sigma0 * (remaining / budget) ** 2 + 1e-8
                offspring = best_x + rng.normal(0, sigma, size=dim)
                offspring = np.clip(offspring, lb, ub)
            # Evaluate
            val = func(offspring)
            calls += 1
            if val < best_val:
                best_val = val
                best_x = offspring.copy()
                report_best(best_val, best_x)
            # Replace worst if offspring is better
            worst_idx = np.argmax(pop_fitness)
            if val < pop_fitness[worst_idx]:
                pop[worst_idx] = offspring
                pop_fitness[worst_idx] = val
        return best_val, best_x