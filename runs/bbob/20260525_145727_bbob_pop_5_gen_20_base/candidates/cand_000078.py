import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        # population size: adaptive, at least 2
        pop_size = max(2, min(10*dim, budget // 2))
        if pop_size > budget:
            pop_size = budget

        # initialize positions
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        # initialize velocities to zero
        vel = np.zeros((pop_size, dim))

        # personal bests
        pbest = pop.copy()
        pbest_val = np.full(pop_size, np.inf)

        # evaluate initial population
        evals = 0
        best_val = np.inf
        best_x = None
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            pbest_val[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # PSO parameters
        w = 0.7
        c1 = 1.5
        c2 = 1.5

        # main loop
        iteration = 0
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # random coefficients
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                # update velocity
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (best_x - pop[i])
                # update position
                pop[i] = pop[i] + vel[i]
                # clip to bounds
                pop[i] = np.clip(pop[i], lb, ub)
                # evaluate
                val = func(pop[i])
                evals += 1
                # update personal best
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = pop[i].copy()
                    # update global best if improved
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
            iteration += 1

        return best_val, best_x