import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(5 * dim, budget // 4))
        self.restart_threshold = max(10, 2 * dim)
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 2.0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        if pop_size < 2:
            best_x = rng.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize positions and velocities
        positions = rng.uniform(lb, ub, (pop_size, dim))
        velocities = rng.uniform(-(ub - lb), (ub - lb), (pop_size, dim)) * 0.1
        pbest_positions = positions.copy()
        pbest_values = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(positions[i])
            evals += 1
            pbest_values[i] = val
            if val < best_val:
                best_val = val
                best_x = positions[i].copy()
                report_best(best_val, best_x)

        no_improve = 0

        while evals < budget:
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                                 self.c2 * r2 * (best_x - positions[i]))
                positions[i] = positions[i] + velocities[i]
                # Clip to bounds
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evals += 1
                if val < pbest_values[i]:
                    pbest_values[i] = val
                    pbest_positions[i] = positions[i].copy()
                if val < best_val:
                    best_val = val
                    best_x = positions[i].copy()
                    improved = True
                    report_best(best_val, best_x)

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # Restart if stagnation
            if no_improve >= self.restart_threshold:
                new_positions = rng.uniform(lb, ub, (pop_size, dim))
                new_positions[0] = best_x.copy()
                new_velocities = rng.uniform(-(ub - lb), (ub - lb), (pop_size, dim)) * 0.1
                new_velocities[0] = 0
                new_pbest_positions = new_positions.copy()
                new_pbest_values = np.full(pop_size, np.inf)
                new_pbest_values[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    val = func(new_positions[i])
                    evals += 1
                    new_pbest_values[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_positions[i].copy()
                        report_best(best_val, best_x)
                positions = new_positions
                velocities = new_velocities
                pbest_positions = new_pbest_positions
                pbest_values = new_pbest_values
                no_improve = 0

        return best_val, best_x