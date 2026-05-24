import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        popsize = min(budget, max(4, min(4 * dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            val = func(pop[i])
            pop_fitness[i] = val
            evals += 1
            if val < self.best_value:
                self.best_value = val
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        F = 0.8
        CR = 0.9

        while evals < budget:
            for i in range(popsize):
                # select three distinct random indices different from i
                indices = list(range(popsize))
                indices.remove(i)
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val <= pop_fitness[i]:
                    pop_fitness[i] = val
                    pop[i] = trial
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
            if evals >= budget:
                break
        return self.best_value, self.best_x