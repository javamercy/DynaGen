import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(10 * dim, budget // 2))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        rng = self.rng

        # Initialize positions and velocities
        positions = rng.uniform(lb, ub, size=(n, dim))
        velocities = rng.uniform(-1, 1, size=(n, dim)) * (ub - lb) * 0.1

        # Evaluate initial swarm
        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0

        for i in range(n):
            if evaluations >= self.budget:
                break
            x = positions[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Personal bests
        pbest_pos = positions.copy()
        pbest_val = fitness.copy()

        if evaluations >= self.budget:
            return best_val, best_x

        # Swarm parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0

        # Iterate until budget exhausted
        while evaluations < self.budget:
            # Linear inertia reduction
            max_iters = self.budget // n
            if max_iters > 0:
                current_iter = evaluations // n
                w = w_start - (w_start - w_end) * current_iter / max_iters
            else:
                w = w_end

            # For each particle
            for i in range(n):
                if evaluations >= self.budget:
                    break

                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                cognitive = c1 * r1 * (pbest_pos[i] - positions[i])
                social = c2 * r2 * (best_x - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate
                val = func(positions[i])
                evaluations += 1

                # Update personal best
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = positions[i].copy()

                # Update global best
                if val < best_val:
                    best_val = val
                    best_x = positions[i].copy()
                    report_best(best_val, best_x)

        return best_val, best_x