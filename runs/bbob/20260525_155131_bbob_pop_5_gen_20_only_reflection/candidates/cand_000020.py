import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(10, min(int(budget / 10), 5 * dim))
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.stagnation_limit = max(5 * dim, 20)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        span = ub - lb
        pop_size = self.pop_size
        # Initialize
        positions = lb + self.rng.rand(pop_size, dim) * span
        velocities = self.rng.uniform(-1, 1, (pop_size, dim)) * span * 0.1
        personal_best_pos = positions.copy()
        personal_best_val = np.full(pop_size, np.inf)
        global_best_val = np.inf
        global_best_pos = np.zeros(dim)
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            val = func(positions[i])
            evals += 1
            personal_best_val[i] = val
            personal_best_pos[i] = positions[i].copy()
            if val < global_best_val:
                global_best_val = val
                global_best_pos = positions[i].copy()
                report_best(global_best_val, global_best_pos)
        if evals >= self.budget:
            return global_best_val, global_best_pos
        no_improve_evals = 0
        while evals < self.budget:
            prev_global = global_best_val
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                r1 = self.rng.rand(dim)
                r2 = self.rng.rand(dim)
                velocities[i] = self.w * velocities[i] + \
                    self.c1 * r1 * (personal_best_pos[i] - positions[i]) + \
                    self.c2 * r2 * (global_best_pos - positions[i])
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evals += 1
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = positions[i].copy()
                    if val < global_best_val:
                        global_best_val = val
                        global_best_pos = positions[i].copy()
                        report_best(global_best_val, global_best_pos)
            if evals >= self.budget:
                break
            if global_best_val < prev_global:
                no_improve_evals = 0
            else:
                no_improve_evals += 1
            if no_improve_evals >= self.stagnation_limit:
                # Restart: reinitialize positions and velocities, keep global best
                positions = lb + self.rng.rand(pop_size, dim) * span
                velocities = self.rng.uniform(-1, 1, (pop_size, dim)) * span * 0.1
                personal_best_pos = positions.copy()
                personal_best_val = np.full(pop_size, np.inf)
                no_improve_evals = 0
        return global_best_val, global_best_pos