import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(10 * dim, budget // 2))
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize

        # Initialize population and velocities
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        vel = np.zeros((popsize, dim))
        pbest = pop.copy()
        pbest_fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0

        # Evaluate initial population
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            pbest_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Main PSO loop
        while evaluations < self.budget:
            w = self.w_start - (self.w_start - self.w_end) * (evaluations / self.budget)
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                vel[i] = w * vel[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (best_x - pop[i])
                new_x = np.clip(pop[i] + vel[i], lb, ub)
                val = func(new_x)
                evaluations += 1
                pop[i] = new_x
                if val < pbest_fitness[i]:
                    pbest_fitness[i] = val
                    pbest[i] = new_x.copy()
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)

        return best_val, best_x