import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        # Population size
        NP = max(4, min(50, budget // (dim + 1)))
        if NP < 4:
            NP = 4

        # Initialize population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        pop_fitness = np.full(NP, np.inf)
        calls = 0
        best_x = None
        best_val = np.inf

        # Evaluate initial population
        for i in range(NP):
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

        # If no evaluations left, return
        if calls >= budget:
            return best_val, best_x

        F = 0.8
        CR = 0.9
        generation = 0
        prev_best_val = best_val

        # Main DE loop
        while calls < budget:
            generation += 1
            improved_this_gen = False
            for i in range(NP):
                if calls >= budget:
                    break
                # Mutation: select three distinct indices different from i
                idxs = [j for j in range(NP) if j != i]
                a, b, c = rng.choice(idxs, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.uniform() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                # Evaluate trial
                val = func(trial)
                calls += 1
                # Selection
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved_this_gen = True
            # Local search after generation if no improvement
            if not improved_this_gen and calls < budget:
                remaining = budget - calls
                sigma = max(1e-3, (remaining / budget) ** 2 * (ub - lb).mean() / 5)
                # Generate a perturbed point from best
                x = best_x + rng.normal(0, sigma, size=dim)
                x = np.clip(x, lb, ub)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            # Also, occasionally do a global random restart (like parent 1)
            if rng.uniform() < 0.05 and calls < budget:
                x = rng.uniform(lb, ub)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

        return best_val, best_x