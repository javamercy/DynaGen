import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        fcalls = 0

        # population size
        pop_size = max(10, min(5 * dim, 100))
        if pop_size > budget:
            pop_size = budget

        # initialize positions and velocities
        positions = self.rng.uniform(lb, ub, (pop_size, dim))
        velocities = np.zeros((pop_size, dim))
        personal_best_pos = positions.copy()
        personal_best_val = np.full(pop_size, np.inf)
        global_best_val = np.inf
        global_best_pos = np.empty(dim)

        # evaluate initial population
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(positions[i], lb, ub)
            val = func(x)
            fcalls += 1
            personal_best_val[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_pos = x.copy()
                report_best(global_best_val, global_best_pos)

        # PSO parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        max_iter = (budget - fcalls) // pop_size  # approximate
        iter_count = 0

        # main loop
        while fcalls < budget:
            # update inertia weight
            if max_iter > 0:
                w = w_start - (w_start - w_end) * min(iter_count, max_iter) / max_iter
            else:
                w = w_end
            iter_count += 1

            for i in range(pop_size):
                if fcalls >= budget:
                    break
                r1 = self.rng.rand(dim)
                r2 = self.rng.rand(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_pos[i] - positions[i]) +
                                 c2 * r2 * (global_best_pos - positions[i]))
                # update position
                positions[i] = positions[i] + velocities[i]
                # clip to bounds
                for j in range(dim):
                    if positions[i, j] < lb[j]:
                        positions[i, j] = lb[j]
                        velocities[i, j] = 0.0
                    elif positions[i, j] > ub[j]:
                        positions[i, j] = ub[j]
                        velocities[i, j] = 0.0

                x = positions[i].copy()
                val = func(x)
                fcalls += 1
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = x
                    if val < global_best_val:
                        global_best_val = val
                        global_best_pos = x.copy()
                        report_best(global_best_val, global_best_pos)

        return global_best_val, global_best_pos