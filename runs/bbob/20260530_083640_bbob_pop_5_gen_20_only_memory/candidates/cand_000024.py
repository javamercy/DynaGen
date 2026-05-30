import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Population size
        pop_size = max(5, min(10 * dim, budget // 2))
        if pop_size < 2:
            pop_size = 2

        # Initialize positions and velocities
        positions = rng.uniform(lb, ub, size=(pop_size, dim))
        velocities = rng.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), size=(pop_size, dim))

        personal_best_pos = positions.copy()
        personal_best_val = np.full(pop_size, np.inf)
        global_best_val = np.inf
        global_best_pos = None
        evals = 0

        # Initial evaluation
        for i in range(pop_size):
            x = np.clip(positions[i], lb, ub)
            val = func(x)
            evals += 1
            personal_best_val[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_pos = x.copy()
                report_best(global_best_val, global_best_pos)
            if evals >= budget:
                return global_best_val, global_best_pos

        # PSO parameters
        inertia = 0.7
        c1 = 1.5
        c2 = 1.5
        max_iter = (budget - evals) // pop_size
        if max_iter < 1:
            max_iter = 1

        for gen in range(max_iter):
            if evals >= budget:
                break
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Update velocity
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                velocities[i] = (inertia * velocities[i] +
                                 c1 * r1 * (personal_best_pos[i] - positions[i]) +
                                 c2 * r2 * (global_best_pos - positions[i]))
                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                # Evaluate
                val = func(positions[i])
                evals += 1
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = positions[i].copy()
                    if val < global_best_val:
                        global_best_val = val
                        global_best_pos = positions[i].copy()
                        report_best(global_best_val, global_best_pos)
                if evals >= budget:
                    break
            if evals >= budget:
                break

        # Final local search using remaining budget
        remaining = budget - evals
        if remaining > 0:
            for _ in range(remaining):
                x = global_best_pos + 0.01 * (ub - lb) * rng.randn(dim)
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                if val < global_best_val:
                    global_best_val = val
                    global_best_pos = x.copy()
                    report_best(global_best_val, global_best_pos)
                if evals >= budget:
                    break

        return global_best_val, global_best_pos