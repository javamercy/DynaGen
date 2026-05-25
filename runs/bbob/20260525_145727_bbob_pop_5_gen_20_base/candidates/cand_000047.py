import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(2, min(30, budget // 2))

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        rng = self.rng
        budget = self.budget

        positions = lb + rng.rand(pop_size, dim) * (ub - lb)
        velocities = rng.randn(pop_size, dim) * 0.1 * (ub - lb)
        personal_best_positions = positions.copy()
        personal_best_values = np.full(pop_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(positions[i])
            evals += 1
            personal_best_values[i] = val
            if val < global_best_value:
                global_best_value = val
                global_best_position = positions[i].copy()
                report_best(global_best_value, global_best_position)

        w = 0.7
        c1 = 1.5
        c2 = 1.5
        max_velocity = 0.2 * (ub - lb)

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 c2 * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evals += 1
                if val < personal_best_values[i]:
                    personal_best_values[i] = val
                    personal_best_positions[i] = positions[i].copy()
                if val < global_best_value:
                    global_best_value = val
                    global_best_position = positions[i].copy()
                    report_best(global_best_value, global_best_position)

        return global_best_value, global_best_position