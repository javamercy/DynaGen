import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = min(5 * dim, max(10, budget // 4))
        pop_size = max(pop_size, 2)
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        velocities = np.random.uniform(-0.2 * (ub - lb), 0.2 * (ub - lb), (pop_size, dim))
        pbest_positions = positions.copy()
        pbest_values = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        for i in range(pop_size):
            x = np.clip(positions[i], lb, ub)
            val = func(x)
            fcalls += 1
            pbest_values[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        w_max = 0.9
        w_min = 0.4
        c1 = 2.0
        c2 = 2.0
        max_velocity = 0.2 * (ub - lb)
        generation = 0
        while fcalls < budget:
            w = w_max - (w_max - w_min) * generation / (budget // pop_size + 1)
            generation += 1
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (best_x - positions[i])
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                fcalls += 1
                if val < pbest_values[i]:
                    pbest_values[i] = val
                    pbest_positions[i] = positions[i].copy()
                    if val < best_f:
                        best_f = val
                        best_x = positions[i].copy()
                        report_best(best_f, best_x)
        return best_f, best_x