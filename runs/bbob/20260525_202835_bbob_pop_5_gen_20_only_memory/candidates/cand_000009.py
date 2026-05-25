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
        popsize = max(4, min(10 * dim, budget // 2))
        if popsize > budget:
            popsize = budget
        # initialize population
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        for i in range(popsize):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # step size adaptation
        sigma = 0.2 * (ub - lb).mean()
        while evals < budget:
            success = 0
            for i in range(popsize):
                if evals >= budget:
                    break
                # generate trial
                trial = pop[i] + sigma * self.rng.normal(0, 1, dim) * (ub - lb)
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    success += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # adapt sigma using 1/5 rule
            if evals >= budget:
                break
            rate = success / popsize
            if rate > 0.2:
                sigma *= np.exp(1.0 / dim)
            else:
                sigma *= np.exp(-1.0 / dim)
        return best_val, best_x