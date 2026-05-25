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

        # Small population for exploitation
        NP = max(4, min(20, budget // (dim + 1)))
        if NP < 4:
            NP = 4

        # Initialize population uniformly
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

        # DE parameters
        F_start = 0.9
        F_end = 0.5
        CR = 0.9
        max_generations = (budget - calls) // NP if NP > 0 else 0
        generation = 0

        while calls < budget:
            # Adaptive F: linear decrease
            if max_generations > 1:
                F = F_start - (F_start - F_end) * (generation / (max_generations - 1))
            else:
                F = F_start
            generation += 1

            for i in range(NP):
                if calls >= budget:
                    break
                # Mutation: current-to-best/1
                r1, r2 = rng.choice([j for j in range(NP) if j != i], size=2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.uniform() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Clip
                trial = np.clip(trial, lb, ub)
                # Evaluate
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

            # Update max_generations for next iteration
            remaining = budget - calls
            if NP > 0:
                max_generations = max(0, remaining // NP)

        return best_val, best_x