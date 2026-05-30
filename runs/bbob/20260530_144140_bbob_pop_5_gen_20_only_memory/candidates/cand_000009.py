import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Population size: at least 4, at most budget//2, and capped at 10*dim
        self.NP = max(4, min(10*dim, budget // 2))
        # Crossover rate and mutation factor
        self.CR = 0.9
        self.F = 0.5

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub

        # Initialization: uniform random inside bounds
        pop = rng.uniform(bounds_lb, bounds_ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_idx = -1
        best_val = np.inf
        best_x = None

        # Evaluate initial population
        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], bounds_lb, bounds_ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            pop[i] = x

        # Evolution loop
        while budget > 0 and NP > 1:
            for i in range(NP):
                if budget <= 0:
                    break
                # Choose two distinct indices different from i
                candidates = [j for j in range(NP) if j != i]
                if len(candidates) < 2:
                    break
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                # Mutation: DE/best/1
                mutant = best_x + self.F * (pop[r1] - pop[r2])
                # Clip to bounds
                mutant = np.clip(mutant, bounds_lb, bounds_ub)
                # Crossover (binomial)
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]  # ensure at least one dimension from mutant
                # Evaluate trial
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Ensure best_x is defined
        if best_x is None:
            x = rng.uniform(bounds_lb, bounds_ub)
            best_val = func(x)
            best_x = x
        return best_val, best_x