import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        # population size
        pop_size = min(10 * dim, budget // 2)
        if pop_size < 2:
            pop_size = 2

        # initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        calls = 0

        # evaluate initial population
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

        # main DE loop
        while calls < budget:
            for i in range(pop_size):
                if calls >= budget:
                    break
                # select three distinct random indices not equal to i
                candidates = [j for j in range(pop_size) if j != i]
                r = rng.choice(candidates, size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
                # mutation
                mutant = a + 0.8 * (b - c)
                # crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < 0.9 or j == j_rand:
                        trial[j] = mutant[j]
                # clip to bounds
                trial = np.clip(trial, lb, ub)
                # evaluate
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