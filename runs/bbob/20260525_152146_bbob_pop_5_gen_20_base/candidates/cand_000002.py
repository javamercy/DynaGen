import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # population size: 10*dim but at most budget/2 and at least 4
        pop_size = max(4, min(10 * dim, budget // 2))
        # initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
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
                from optimizer_api import report_best
                report_best(best_val, best_x)

        # main DE loop
        while calls < budget:
            for i in range(pop_size):
                if calls >= budget:
                    break
                # select three distinct random indices != i
                candidates = list(range(pop_size))
                candidates.remove(i)
                idxs = rng.choice(candidates, size=3, replace=False)
                a, b, c = idxs
                # mutation: DE/rand/1
                F = 0.8
                mutant = pop[a] + F * (pop[b] - pop[c])
                # clip to bounds
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                Cr = 0.9
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < Cr or j == j_rand:
                        trial[j] = mutant[j]
                # evaluate trial
                trial_val = func(trial)
                calls += 1
                if trial_val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_val
                    if trial_val < best_val:
                        best_val = trial_val
                        best_x = trial.copy()
                        from optimizer_api import report_best
                        report_best(best_val, best_x)
            # optionally adapt parameters (not implemented for simplicity)
        return best_val, best_x