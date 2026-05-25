import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(10, min(2 * dim, budget // 10))
        self.restart_threshold = max(10, dim)

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
        personal_best_pos = positions.copy()
        personal_best_val = np.full(pop_size, np.inf)
        global_best_val = np.inf
        global_best_pos = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            x = positions[i]
            val = func(x)
            evals += 1
            personal_best_val[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_pos = x.copy()
                report_best(global_best_val, global_best_pos)

        max_iter = (budget - evals) // pop_size if pop_size > 0 else 0
        no_improve = 0
        gen = 0
        w_start = 0.9
        w_end = 0.4

        while evals < budget and gen < max_iter:
            w = w_start - (w_start - w_end) * (gen / max_iter) if max_iter > 0 else w_end
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                velocities[i] = w * velocities[i] + 2.0 * r1 * (personal_best_pos[i] - positions[i]) + 2.0 * r2 * (global_best_pos - positions[i])
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evals += 1
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = positions[i].copy()
                    improved = True
                    if val < global_best_val:
                        global_best_val = val
                        global_best_pos = positions[i].copy()
                        report_best(global_best_val, global_best_pos)

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold and evals < budget:
                # Restart: reinitialize all except global best
                new_positions = rng.uniform(lb, ub, (pop_size, dim))
                new_velocities = rng.uniform(-(ub - lb), (ub - lb), (pop_size, dim)) * 0.1
                new_personal_best_val = np.full(pop_size, np.inf)
                # Keep global best
                new_positions[0] = global_best_pos.copy()
                new_velocities[0] = 0
                new_personal_best_val[0] = global_best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_positions[i]
                    val = func(x)
                    evals += 1
                    new_personal_best_val[i] = val
                    if val < global_best_val:
                        global_best_val = val
                        global_best_pos = x.copy()
                        report_best(global_best_val, global_best_pos)
                positions = new_positions
                velocities = new_velocities
                personal_best_pos = new_positions.copy()
                personal_best_val = new_personal_best_val
                no_improve = 0
            gen += 1

        return global_best_val, global_best_pos