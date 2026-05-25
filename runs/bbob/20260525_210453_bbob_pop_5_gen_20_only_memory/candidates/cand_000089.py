import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
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

        if budget < 1:
            best_x = np.full(dim, np.nan)
            best_f = np.inf
            return best_f, best_x

        pop_size = min(5 * dim, max(10, budget // 4))
        if pop_size < 1:
            pop_size = 1

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0

        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)

        if best_x is None:
            best_x = np.clip(np.random.uniform(lb, ub), lb, ub)
            best_f = func(best_x)
            fcalls += 1
            report_best(best_f, best_x)

        F = 0.8
        CR = 0.9

        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break

                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) >= 3:
                    r1, r2, r3 = random.sample(candidates, 3)
                else:
                    r1 = r2 = r3 = i

                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)

                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

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