import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(20, min(4 * dim, budget // 2))
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        # Initialize positions and velocities
        positions = self.rng.uniform(lb, ub, size=(popsize, dim))
        velocities = self.rng.uniform(-(ub - lb), (ub - lb), size=(popsize, dim)) * 0.1
        personal_best_pos = positions.copy()
        personal_best_val = np.full(popsize, np.inf)
        global_best_pos = None
        global_best_val = np.inf
        evaluations = 0

        # Initial evaluation
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            x = positions[i]
            val = func(x)
            evaluations += 1
            personal_best_val[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_pos = x.copy()
                report_best(global_best_val, global_best_pos)

        if global_best_pos is None:
            x = self.rng.uniform(lb, ub)
            val = func(x)
            evaluations += 1
            global_best_val = val
            global_best_pos = x.copy()
            report_best(global_best_val, global_best_pos)

        # PSO iterations
        max_iter = (self.budget - evaluations) // popsize if popsize > 0 else 0
        for gen in range(max_iter):
            if evaluations >= self.budget:
                break
            w = self.w_start - (self.w_start - self.w_end) * (gen / max_iter)
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                # Update velocity
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_pos[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_pos - positions[i]))
                # Clamp velocity to avoid too large steps
                velocities[i] = np.clip(velocities[i], -(ub - lb), (ub - lb))
                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                # Evaluate
                val = func(positions[i])
                evaluations += 1
                # Update personal best
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = positions[i].copy()
                # Update global best
                if val < global_best_val:
                    global_best_val = val
                    global_best_pos = positions[i].copy()
                    report_best(global_best_val, global_best_pos)

        return global_best_val, global_best_pos