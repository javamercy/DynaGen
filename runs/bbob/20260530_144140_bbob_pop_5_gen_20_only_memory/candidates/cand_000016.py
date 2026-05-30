import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.NP = max(4, min(10*dim, budget // 2))
        self.CR = 0.9
        self.F = 0.5

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialization
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

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

        # Evolution
        while budget > 0 and NP > 1:
            for i in range(NP):
                if budget <= 0:
                    break
                candidates = [j for j in range(NP) if j != i]
                if len(candidates) < 3:
                    break
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                # DE/rand/1 mutation
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        if best_x is None:
            x = rng.uniform(lb, ub)
            best_val = func(x)
            best_x = x
        return best_val, best_x