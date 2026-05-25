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
        rng = self.rng
        budget = self.budget

        # population size
        pop_size = max(4, min(10*dim, budget // 4))
        if pop_size > budget:
            pop_size = budget
        if pop_size < 2:
            pop_size = budget

        # initialize particles
        pos = lb + rng.rand(pop_size, dim) * (ub - lb)
        vel = rng.randn(pop_size, dim) * 0.1 * (ub - lb)  # small initial velocities
        # personal bests
        pbest_pos = pos.copy()
        pbest_val = np.full(pop_size, np.inf)
        # global best
        gbest_pos = None
        gbest_val = np.inf
        evals = 0

        # initial evaluation
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pos[i])
            evals += 1
            pbest_val[i] = val
            pbest_pos[i] = pos[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = pos[i].copy()
                report_best(gbest_val, gbest_pos)

        if evals == 0:
            return gbest_val, gbest_pos

        # PSO parameters
        c1 = 2.0
        c2 = 2.0
        w_max = 0.9
        w_min = 0.4
        # budget for PSO phase
        budget_pso = int(0.8 * budget)
        if budget_pso < pop_size:
            budget_pso = budget

        # main PSO loop
        while evals < budget_pso and evals < budget:
            # compute progress for inertia weight
            fraction = evals / budget_pso
            w = w_max - (w_max - w_min) * fraction
            for i in range(pop_size):
                if evals >= budget_pso or evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                # velocity update
                vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
                # optionally clamp velocity (not necessary but helpful)
                max_vel = 0.2 * (ub - lb)
                vel[i] = np.clip(vel[i], -max_vel, max_vel)
                # position update and clipping
                pos[i] = pos[i] + vel[i]
                pos[i] = np.clip(pos[i], lb, ub)
                # evaluate
                val = func(pos[i])
                evals += 1
                # update personal best if improvement
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = pos[i].copy()
                    # update global best if improvement
                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = pos[i].copy()
                        report_best(gbest_val, gbest_pos)

        # local refinement around global best
        remaining = budget - evals
        if remaining > 0 and gbest_pos is not None:
            sigma = 0.1 * (ub - lb)
            for _ in range(remaining):
                perturb = rng.normal(0, sigma, dim)
                candidate = gbest_pos + perturb
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = candidate.copy()
                    report_best(gbest_val, gbest_pos)

        return gbest_val, gbest_pos