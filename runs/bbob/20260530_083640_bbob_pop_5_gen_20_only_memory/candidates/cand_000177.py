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

        # Population size: scaled by dimension, limited by budget
        pop_size = max(1, min(max(3*dim, 5), budget // 5, 20))
        pop_size = min(pop_size, budget)  # ensure not exceed budget

        # Initialize positions and velocities
        pos = lb + rng.rand(pop_size, dim) * (ub - lb)
        vel = np.zeros_like(pos)
        v_max = 0.2 * (ub - lb)

        # Evaluate initial population
        best_f = np.inf
        best_x = None
        pbest_f = np.full(pop_size, np.inf)
        pbest_pos = pos.copy()

        for i in range(pop_size):
            x = np.clip(pos[i], lb, ub)
            f = func(x)
            if f < pbest_f[i]:
                pbest_f[i] = f
                pbest_pos[i] = x.copy()
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        evals = pop_size

        if evals >= budget:
            return best_f, best_x

        # PSO parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0

        # Estimate number of generations for inertia scheduling
        max_gen = (budget - evals) // pop_size
        gen = 0

        while evals < budget:
            w = w_start - (w_start - w_end) * (gen / max_gen) if max_gen > 0 else w_end
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (best_x - pos[i])
                vel[i] = np.clip(vel[i], -v_max, v_max)
                pos[i] = np.clip(pos[i] + vel[i], lb, ub)
                f = func(pos[i])
                evals += 1
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_pos[i] = pos[i].copy()
                if f < best_f:
                    best_f = f
                    best_x = pos[i].copy()
                    report_best(best_f, best_x)
            gen += 1

        return best_f, best_x