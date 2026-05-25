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
        # Larger population size
        npop = max(10, min(30, budget // (dim * 2)))
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
        # Adaptation parameters
        sigma = 0.2 * np.mean(ub - lb)
        # Main loop
        while calls < budget:
            # Generate offspring
            if rng.uniform() < 0.5:
                offspring = rng.uniform(lb, ub)
            else:
                # Mutation of a random parent (not necessarily best)
                idx = rng.integers(npop)
                parent = pop[idx]
                # Step size based on population spread
                spread = np.std(pop, axis=0)
                step = sigma * spread + 1e-8
                offspring = parent + rng.normal(0, step, size=dim)
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
            # Periodic restart of worst half every 20% of budget
            if calls % max(1, budget // 5) == 0 and calls < budget:
                # Sort by fitness
                sorted_idx = np.argsort(pop_fitness)
                # Keep best half, replace worst half with random points
                for j in range(npop // 2, npop):
                    pop[sorted_idx[j]] = rng.uniform(lb, ub)
                    x_eval = pop[sorted_idx[j]]
                    val = func(x_eval)
                    calls += 1
                    pop_fitness[sorted_idx[j]] = val
                    if val < best_val:
                        best_val = val
                        best_x = x_eval.copy()
                        report_best(best_val, best_x)
                    if calls >= budget:
                        break
        return best_val, best_x