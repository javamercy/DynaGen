import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Smaller population for more generations
        self.NP = max(4, min(5*dim, budget // 4))
        self.F = 0.3
        self.CR = 0.95

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_idx = -1
        best_val = np.inf
        best_x = None

        # Evaluate initial population
        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            pop[i] = x

        # DE evolution with current-to-best mutation
        generation = 0
        while budget > 0 and NP > 1:
            for i in range(NP):
                if budget <= 0:
                    break
                # Choose two distinct indices different from i
                candidates = [j for j in range(NP) if j != i]
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                # current-to-best mutation
                mutant = pop[i] + self.F * (best_x - pop[i]) + self.F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # evaluate
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            generation += 1

        # Local search around best with remaining budget
        if budget > 0 and best_x is not None:
            sigma_init = 0.1 * (ub - lb)
            total_local = budget
            for k in range(total_local):
                if budget <= 0:
                    break
                sigma = sigma_init * (1 - k / total_local)
                candidate = best_x + rng.normal(0, sigma)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        # Fallback if best_x is None (should not happen)
        if best_x is None:
            best_x = rng.uniform(lb, ub)
            best_val = func(best_x)
            # No report_best because budget might be 0, but we call func anyway; but to be safe, we assume budget>0
        return best_val, best_x