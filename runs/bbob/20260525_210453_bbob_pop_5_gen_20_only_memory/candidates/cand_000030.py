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
        pop_size = min(5 * dim, max(5, budget // 5))
        pop_size = max(3, min(pop_size, budget // 2))
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
        local_budget = int(0.2 * budget)
        DE_budget = budget - local_budget
        while fcalls < DE_budget:
            for i in range(pop_size):
                if fcalls >= DE_budget:
                    break
                progress = fcalls / budget
                F = 0.9 - 0.7 * progress
                CR = 0.9 - 0.7 * progress
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = random.sample(candidates, 2)
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
        # Pattern search
        step = 0.1 * (ub - lb)
        x = best_x.copy()
        f = best_f
        while fcalls < budget:
            improved = False
            dims = list(range(dim))
            random.shuffle(dims)
            for j in dims:
                if fcalls >= budget:
                    break
                x_new = x.copy()
                x_new[j] = np.clip(x[j] + step[j], lb[j], ub[j])
                val = func(x_new)
                fcalls += 1
                if val < f:
                    f = val
                    x = x_new
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    break
                x_new = x.copy()
                x_new[j] = np.clip(x[j] - step[j], lb[j], ub[j])
                val = func(x_new)
                fcalls += 1
                if val < f:
                    f = val
                    x = x_new
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    break
            if not improved:
                step = step * 0.5
                if np.max(step) < 1e-12:
                    break
        return best_f, best_x