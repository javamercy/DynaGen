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
        diff = ub - lb

        # Reserve budget for Nelder-Mead: at least 2*dim, at most 20% of budget
        nm_budget = max(2 * dim, int(0.2 * budget))
        nm_budget = min(nm_budget, budget - 1)
        pso_budget = budget - nm_budget

        # PSO parameters
        pop_size = min(10 * dim, pso_budget // 2)
        pop_size = max(pop_size, 4)
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        max_vel = 0.2 * diff

        # Initialize particles
        positions = rng.uniform(lb, ub, (pop_size, dim))
        velocities = rng.uniform(-max_vel, max_vel, (pop_size, dim))
        pbest_positions = positions.copy()
        pbest_values = np.full(pop_size, np.inf)
        gbest_value = np.inf
        gbest_position = np.zeros(dim)

        evals = 0
        for i in range(pop_size):
            x = np.clip(positions[i], lb, ub)
            val = func(x)
            evals += 1
            pbest_values[i] = val
            pbest_positions[i] = x
            if val < gbest_value:
                gbest_value = val
                gbest_position = x.copy()
                report_best(gbest_value, gbest_position)

        # PSO main loop
        max_iter = (pso_budget - evals) // pop_size
        if max_iter > 0:
            for gen in range(max_iter):
                w = w_start - (w_start - w_end) * gen / max_iter
                for i in range(pop_size):
                    r1 = rng.rand(dim)
                    r2 = rng.rand(dim)
                    velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (gbest_position - positions[i])
                    # Clamp velocity
                    velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                    positions[i] = positions[i] + velocities[i]
                    positions[i] = np.clip(positions[i], lb, ub)
                # Evaluate all particles
                for i in range(pop_size):
                    if evals >= pso_budget:
                        break
                    x = positions[i]
                    val = func(x)
                    evals += 1
                    if val < pbest_values[i]:
                        pbest_values[i] = val
                        pbest_positions[i] = x.copy()
                        if val < gbest_value:
                            gbest_value = val
                            gbest_position = x.copy()
                            report_best(gbest_value, gbest_position)
                if evals >= pso_budget:
                    break

        # Nelder-Mead local search from gbest
        if evals < budget:
            step = 0.1 * diff
            simplex = np.tile(gbest_position, (dim + 1, 1))
            for i in range(dim):
                simplex[i+1, i] = np.clip(gbest_position[i] + step[i], lb[i], ub[i])
            fvals = np.full(dim + 1, np.inf)
            fvals[0] = gbest_value
            for i in range(1, dim + 1):
                if evals >= budget:
                    break
                x = np.clip(simplex[i], lb, ub)
                val = func(x)
                evals += 1
                fvals[i] = val
                if val < gbest_value:
                    gbest_value = val
                    gbest_position = x.copy()
                    report_best(gbest_value, gbest_position)

            rho = 1.0
            chi = 2.0
            psi = 0.5
            sigma = 0.5

            while evals < budget:
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                best_val_local = fvals[0]
                worst_val = fvals[-1]
                second_worst_val = fvals[-2]
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + rho * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget: break
                fr = func(xr)
                evals += 1
                if fr < best_val_local:
                    # Expansion
                    xe = centroid + chi * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= budget: break
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < gbest_value:
                            gbest_value = fe
                            gbest_position = xe.copy()
                            report_best(gbest_value, gbest_position)
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        if fr < gbest_value:
                            gbest_value = fr
                            gbest_position = xr.copy()
                            report_best(gbest_value, gbest_position)
                elif fr < second_worst_val:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < gbest_value:
                        gbest_value = fr
                        gbest_position = xr.copy()
                        report_best(gbest_value, gbest_position)
                else:
                    if fr < worst_val:
                        # Outside contraction
                        xc = centroid + psi * (xr - centroid)
                        xc = np.clip(xc, lb, ub)
                        if evals >= budget: break
                        fc = func(xc)
                        evals += 1
                        if fc < fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < gbest_value:
                                gbest_value = fc
                                gbest_position = xc.copy()
                                report_best(gbest_value, gbest_position)
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                if evals >= budget: break
                                val_i = func(simplex[i])
                                evals += 1
                                fvals[i] = val_i
                                if val_i < gbest_value:
                                    gbest_value = val_i
                                    gbest_position = simplex[i].copy()
                                    report_best(gbest_value, gbest_position)
                            if evals >= budget: break
                    else:
                        # Inside contraction
                        xc = centroid - psi * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        if evals >= budget: break
                        fc = func(xc)
                        evals += 1
                        if fc < worst_val:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < gbest_value:
                                gbest_value = fc
                                gbest_position = xc.copy()
                                report_best(gbest_value, gbest_position)
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                if evals >= budget: break
                                val_i = func(simplex[i])
                                evals += 1
                                fvals[i] = val_i
                                if val_i < gbest_value:
                                    gbest_value = val_i
                                    gbest_position = simplex[i].copy()
                                    report_best(gbest_value, gbest_position)
                            if evals >= budget: break

        return gbest_value, gbest_position