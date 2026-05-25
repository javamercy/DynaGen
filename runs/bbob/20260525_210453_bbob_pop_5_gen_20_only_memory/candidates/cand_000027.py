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
        pop_size = min(5 * dim, max(10, budget // 4))
        # Initialize positions and velocities
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        velocities = np.random.uniform(-np.abs(ub - lb) * 0.1, np.abs(ub - lb) * 0.1, (pop_size, dim))
        # Personal bests
        pbest_positions = positions.copy()
        pbest_values = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        # Evaluate initial swarm
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(positions[i], lb, ub)
            val = func(x)
            fcalls += 1
            pbest_values[i] = val
            pbest_positions[i] = x.copy()
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        # PSO parameters
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        # Main loop
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                r1 = random.random()
                r2 = random.random()
                # Velocity update
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (best_x - positions[i])
                # Position update
                positions[i] = positions[i] + velocities[i]
                # Clip to bounds
                positions[i] = np.clip(positions[i], lb, ub)
                # Evaluate
                val = func(positions[i])
                fcalls += 1
                # Update personal best
                if val < pbest_values[i]:
                    pbest_values[i] = val
                    pbest_positions[i] = positions[i].copy()
                    # Update global best
                    if val < best_f:
                        best_f = val
                        best_x = positions[i].copy()
                        report_best(best_f, best_x)
        return best_f, best_x