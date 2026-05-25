import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(20, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        dim = self.dim
        rng = self.rng
        budget = self.budget

        # Initialize positions and velocities
        positions = lb + rng.rand(pop_size, dim) * (ub - lb)
        velocities = rng.uniform(-1, 1, (pop_size, dim)) * (ub - lb) * 0.1
        personal_best_positions = positions.copy()
        personal_best_values = np.full(pop_size, np.inf)
        global_best_pos = None
        global_best_val = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(positions[i])
            evals += 1
            personal_best_values[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_pos = positions[i].copy()
                report_best(global_best_val, global_best_pos)

        # PSO parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0

        # Main loop
        while evals < budget:
            w = w_start - (w_start - w_end) * (evals / budget)
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (personal_best_positions[i] - positions[i])
                                 + c2 * r2 * (global_best_pos - positions[i]))
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evals += 1
                if val < personal_best_values[i]:
                    personal_best_values[i] = val
                    personal_best_positions[i] = positions[i].copy()
                if val < global_best_val:
                    global_best_val = val
                    global_best_pos = positions[i].copy()
                    report_best(global_best_val, global_best_pos)

        return global_best_val, global_best_pos