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
            best_x = np.full(dim, np.nan)
            best_f = np.inf
        CR = 0.9
        stagnation = 0
        max_stag = max(pop_size // 2, 5)
        while fcalls < budget:
            improved = False
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) >= 2:
                    r1, r2 = random.sample(candidates, 2)
                else:
                    r1 = r2 = candidates[0] if len(candidates) >= 1 else i
                F = np.random.uniform(0.7, 0.9)
                if best_x is None:
                    best_x = pop[np.argmin(pop_f)].copy()
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
            if improved:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= max_stag and fcalls < budget:
                for i in range(pop_size):
                    if fcalls >= budget:
                        break
                    if pop_f[i] == best_f:
                        continue
                    new_x = np.random.uniform(lb, ub, dim)
                    new_x = np.clip(new_x, lb, ub)
                    val = func(new_x)
                    fcalls += 1
                    pop[i] = new_x
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = new_x.copy()
                        report_best(best_f, best_x)
                stagnation = 0
        return best_f, best_x