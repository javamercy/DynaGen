import numpy as np

def report_best(value, x):
    pass

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

        # population size, at least 4 but not more than budget
        pop_size = max(4, min(50, budget // 2))
        if pop_size > budget:
            pop_size = budget
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        fitness = np.full(pop_size, np.inf)

        best_val = np.inf
        best_x = np.zeros(dim)

        # evaluate initial population
        for i in range(pop_size):
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

        # main DE loop
        while budget > 0:
            for i in range(pop_size):
                if budget <= 0:
                    break
                # choose three distinct indices different from i
                idxs = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(idxs, 3, replace=False)
                # mutation
                F = 0.8
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                CR = 0.9
                cross_mask = rng.rand(dim) < CR
                if not np.any(cross_mask):
                    cross_mask[rng.randint(dim)] = True
                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                # evaluate trial
                trial_val = func(trial)
                budget -= 1
                # selection
                if trial_val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_val
                    if trial_val < best_val:
                        best_val = trial_val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x