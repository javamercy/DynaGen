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
        rng = self.rng
        budget = self.budget
        evals = 0

        # Initial Latin Hypercube Sampling (smaller sample)
        n_init = min(2 * dim, budget // 10)
        if n_init < 2:
            n_init = 2
        points = np.empty((n_init, dim))
        for i in range(dim):
            points[:, i] = rng.uniform(lb[i], ub[i], size=n_init)
        for i in range(dim):
            rng.shuffle(points[:, i])

        best_x = points[0].copy()
        best_f = func(best_x)
        evals += 1
        report_best(best_f, best_x)

        for i in range(1, n_init):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Main local search loop with restarts
        while evals < budget:
            # Build simplex with adaptive step
            step = 0.02 * (ub - lb)
            step = np.where(step == 0, 1e-8, step)
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(dim):
                x = best_x.copy()
                x[i] += step[i]
                x = np.clip(x, lb, ub)
                simplex[i+1] = x

            f_vals = np.full(dim + 1, np.inf)
            for i in range(dim + 1):
                if evals >= budget:
                    break
                f_vals[i] = func(simplex[i])
                evals += 1
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = simplex[i].copy()
                    report_best(best_f, best_x)

            max_iter = min(100, (budget - evals) // (dim + 1) + 1)
            for _ in range(max_iter):
                if evals >= budget:
                    break
                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]

                # Convergence: function range small or simplex size small
                if f_vals[-1] - f_vals[0] < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals += 1
                if evals > budget:
                    break
                if fr < f_vals[0]:
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    evals += 1
                    if fc < f_vals[-1]:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            f_vals[i] = func(simplex[i])
                            evals += 1
                            if evals > budget:
                                break

                idx_min = np.argmin(f_vals)
                if f_vals[idx_min] < best_f:
                    best_f = f_vals[idx_min]
                    best_x = simplex[idx_min].copy()
                    report_best(best_f, best_x)

            # Restart with perturbation
            if evals < budget:
                pert = rng.uniform(-0.1, 0.1, size=dim) * (ub - lb)
                new_x = np.clip(best_x + pert, lb, ub)
                new_f = func(new_x)
                evals += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    report_best(best_f, best_x)

        return best_f, best_x