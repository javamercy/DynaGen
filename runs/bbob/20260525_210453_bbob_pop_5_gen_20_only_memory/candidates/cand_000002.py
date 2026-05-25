import numpy as np
import random

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # population size
        pop_size = min(5 * dim, max(10, budget // 4))
        # ensure at least 2? but fine
        # generate initial population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        # evaluate
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = pop[i]
            # clip to be safe
            x = np.clip(x, lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        # remaining evaluations
        F = 0.8
        CR = 0.9
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # mutation: select three distinct random indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                r0, r1, r2 = random.sample(candidates, 3)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                # clip mutant to bounds
                mutant = np.clip(mutant, lb, ub)
                # crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # evaluate trial
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
        return best_f, best_x