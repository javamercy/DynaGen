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
        # Determine population size
        pop_size = min(5 * dim, max(10, budget // 2))
        if pop_size < 2 or pop_size > budget:
            # Fallback to random search
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                    report_best(best_f, best_x)
            return best_f, best_x

        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        vel = np.zeros((pop_size, dim))
        pbest = pop.copy()
        pbest_f = np.full(pop_size, np.inf)
        gbest = None
        gbest_f = np.inf
        fcalls = 0

        # Evaluate initial population
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pbest_f[i] = val
            pbest[i] = x.copy()
            if val < gbest_f:
                gbest_f = val
                gbest = x.copy()
                report_best(gbest_f, gbest)

        # PSO parameters
        c1 = 2.0
        c2 = 2.0
        w_start = 0.9
        w_end = 0.4
        max_iter = (budget - fcalls) // pop_size  # number of full iterations
        if max_iter <= 0:
            # If budget exhausted after initial evaluation, return best
            return gbest_f, gbest

        it = 0
        while fcalls < budget and it < max_iter:
            w = w_start - (w_start - w_end) * it / (max_iter - 1) if max_iter > 1 else w_start
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                r1 = np.random.uniform(size=dim)
                r2 = np.random.uniform(size=dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
                # Optional: velocity clamping (not necessary due to clipping)
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                val = func(pop[i])
                fcalls += 1
                if val < pbest_f[i]:
                    pbest_f[i] = val
                    pbest[i] = pop[i].copy()
                    if val < gbest_f:
                        gbest_f = val
                        gbest = pop[i].copy()
                        report_best(gbest_f, gbest)
            it += 1

        return gbest_f, gbest