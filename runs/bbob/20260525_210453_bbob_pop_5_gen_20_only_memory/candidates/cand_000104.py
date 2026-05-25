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
        pop_size = min(10 * dim, max(10, budget // 2))
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        vel = np.random.uniform(-(ub - lb), (ub - lb), (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        personal_best = pop.copy()
        personal_best_f = np.full(pop_size, np.inf)
        fcalls = 0
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            personal_best_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        max_iter = budget // pop_size
        iteration = 0
        while fcalls < budget:
            w = w_start - (w_start - w_end) * (iteration / max_iter) if max_iter > 0 else w_end
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                r1 = np.random.uniform(0, 1, dim)
                r2 = np.random.uniform(0, 1, dim)
                vel[i] = w * vel[i] + c1 * r1 * (personal_best[i] - pop[i]) + c2 * r2 * (best_x - pop[i])
                pop[i] = pop[i] + vel[i]
                for j in range(dim):
                    if pop[i][j] < lb[j]:
                        pop[i][j] = lb[j]
                        vel[i][j] *= -0.5
                    elif pop[i][j] > ub[j]:
                        pop[i][j] = ub[j]
                        vel[i][j] *= -0.5
                val = func(pop[i])
                fcalls += 1
                if val < personal_best_f[i]:
                    personal_best_f[i] = val
                    personal_best[i] = pop[i].copy()
                    if val < best_f:
                        best_f = val
                        best_x = pop[i].copy()
                        report_best(best_f, best_x)
            iteration += 1
        return best_f, best_x