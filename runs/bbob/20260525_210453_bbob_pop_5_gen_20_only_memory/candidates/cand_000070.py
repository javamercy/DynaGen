import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        # Population size: moderate, adjusting to budget
        pop_size = min(10 * dim, max(20, budget // 10))
        if pop_size < 2:
            pop_size = 2

        # Initialize positions and velocities
        pop = rng.uniform(lb, ub, (pop_size, dim))
        vel = np.zeros((pop_size, dim))

        # Personal bests
        pbest = pop.copy()
        pbest_f = np.full(pop_size, np.inf)

        # Global best
        gbest = None
        gbest_f = np.inf

        fcalls = 0

        # Initial evaluations
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
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        v_max = 0.2 * (ub - lb)  # velocity clamping

        # Main loop
        generation = 0
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Update velocity
                r1 = rng.uniform(0, 1, dim)
                r2 = rng.uniform(0, 1, dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
                # Clamp velocity
                vel[i] = np.clip(vel[i], -v_max, v_max)
                # Update position
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                # Evaluate
                val = func(pop[i])
                fcalls += 1
                if val < pbest_f[i]:
                    pbest_f[i] = val
                    pbest[i] = pop[i].copy()
                    if val < gbest_f:
                        gbest_f = val
                        gbest = pop[i].copy()
                        report_best(gbest_f, gbest)
            generation += 1

        return gbest_f, gbest