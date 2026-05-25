import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(4 * dim, budget // 2))
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.v_max_factor = 0.2

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        rng = self.rng

        positions = rng.uniform(lb, ub, size=(popsize, dim))
        velocities = rng.uniform(-0.1*(ub-lb), 0.1*(ub-lb), size=(popsize, dim))
        pbest_positions = positions.copy()
        pbest_values = np.full(popsize, np.inf)
        gbest_position = None
        gbest_value = np.inf

        evaluations = 0
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            x = positions[i]
            val = func(x)
            evaluations += 1
            pbest_values[i] = val
            if val < gbest_value:
                gbest_value = val
                gbest_position = x.copy()
                report_best(gbest_value, gbest_position)

        iteration = 0
        while evaluations < self.budget:
            w = self.w_start - (self.w_start - self.w_end) * iteration / (self.budget // popsize + 1)
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                                 self.c2 * r2 * (gbest_position - positions[i]))
                v_max = self.v_max_factor * (ub - lb)
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evaluations += 1
                if val < pbest_values[i]:
                    pbest_values[i] = val
                    pbest_positions[i] = positions[i].copy()
                    if val < gbest_value:
                        gbest_value = val
                        gbest_position = positions[i].copy()
                        report_best(gbest_value, gbest_position)
            iteration += 1
        return gbest_value, gbest_position