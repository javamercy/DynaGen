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
        # smaller population for more generations
        pop_size = max(4, min(20, 2 * dim))
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
        F_start = 0.9
        F_end = 0.4
        CR = 0.9
        while fcalls < budget:
            F = F_start - (F_start - F_end) * (fcalls / budget)
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) >= 2:
                    r1, r2 = random.sample(candidates, 2)
                else:
                    r1 = r2 = 0
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                trial = pop[i].copy()
                j_rand = random.randint(0, dim - 1)
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
            # local search around best
            if fcalls < budget:
                sigma = 0.05 * (ub - lb) * (1 - fcalls / budget)
                num_local = min(5, budget - fcalls)
                for _ in range(num_local):
                    candidate = best_x + sigma * np.random.randn(dim)
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    fcalls += 1
                    if val < best_f:
                        best_f = val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
        return best_f, best_x