import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10*dim, budget))
        if self.pop_size > budget:
            self.pop_size = budget
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        rng = self.rng
        budget = self.budget

        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[i] + self.F * (best_x - pop[i]) + self.F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                j_rand = rng.randint(dim)
                mask = rng.rand(dim) < self.CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x