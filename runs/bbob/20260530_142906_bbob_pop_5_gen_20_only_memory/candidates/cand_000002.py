import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Population size
        NP = max(4, min(int(5 * dim), 40, budget // 2))
        # If NP is too large for remaining budget, adjust
        if NP > budget:
            NP = budget

        # Initialization
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        # Evaluate initial population
        for i in range(NP):
            if evals >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fitness[i] = val
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                # report_best is a global helper
                report_best(best_val, best_x)

        # If budget exhausted after initialization, return
        if evals >= budget:
            return best_val, best_x

        # Main DE loop
        while evals < budget:
            # Mutation and crossover parameters
            F = rng.uniform(0.5, 1.0)
            CR = rng.uniform(0.5, 1.0)
            # For each parent
            for i in range(NP):
                if evals >= budget:
                    break
                # Choose three distinct random indices different from i
                idxs = [j for j in range(NP) if j != i]
                r = rng.choice(idxs, size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
                # Mutation
                mutant = a + F * (b - c)
                # Crossover
                j_rand = rng.integers(dim)
                trial = np.array([mutant[j] if rng.random() < CR or j == j_rand else pop[i, j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)
                # Selection
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # If we break mid-generation, we still have some points left in budget
            # but loop condition will handle
        return best_val, best_x