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
        pop_size = min(5 * dim, max(10, budget // 4))
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
        gen = 0
        max_gen = max(1, budget // pop_size)
        F_init = 0.9
        F_final = 0.4
        CR_init = 0.9
        CR_final = 0.5
        plateau = 0
        while fcalls < budget:
            # reduce population size after half budget
            if fcalls > budget / 2 and pop_size > 5:
                pop_size = max(5, pop_size // 2)
                pop = pop[:pop_size]
                pop_f = pop_f[:pop_size]
            F = F_init - (F_init - F_final) * gen / max_gen
            CR = CR_init - (CR_init - CR_final) * gen / max_gen
            improved = False
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = random.sample(candidates, 2) if len(candidates) >= 2 else (candidates[0], candidates[0])
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
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
                        improved = True
            if not improved:
                plateau += 1
            else:
                plateau = 0
            # local search if stagnation
            if plateau >= 3 and fcalls < budget:
                sigma = 0.005 * (ub - lb) * (1 - gen / max_gen)
                x_try = best_x + sigma * np.random.randn(dim)
                x_try = np.clip(x_try, lb, ub)
                val = func(x_try)
                fcalls += 1
                if val < best_f:
                    best_f = val
                    best_x = x_try.copy()
                    report_best(best_f, best_x)
                plateau = 0
            gen += 1
        return best_f, best_x