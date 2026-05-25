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

        pop_size = max(5, min(budget, 4 * dim))
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        max_velocity = 0.5 * (ub - lb)
        velocities = np.random.uniform(-max_velocity, max_velocity, (pop_size, dim))

        personal_best_x = positions.copy()
        personal_best_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0

        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(positions[i], lb, ub)
            val = func(x)
            fcalls += 1
            personal_best_f[i] = val
            personal_best_x[i] = x.copy()
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)

        w = 0.7
        c1 = 1.5
        c2 = 1.5

        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_x[i] - positions[i]) + c2 * r2 * (best_x - positions[i])
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                val = func(positions[i])
                fcalls += 1
                if val < personal_best_f[i]:
                    personal_best_f[i] = val
                    personal_best_x[i] = positions[i].copy()
                    if val < best_f:
                        best_f = val
                        best_x = positions[i].copy()
                        report_best(best_f, best_x)

        return best_f, best_x