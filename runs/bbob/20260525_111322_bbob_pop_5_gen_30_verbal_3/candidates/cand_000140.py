import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        best_x = None
        best_f = np.inf
        evals = 0
        n_vertices = dim + 1

        # Initial random sampling to build simplex
        n_init = min(n_vertices, max(5, int(0.1 * budget)))
        init_points = []
        init_vals = []
        for _ in range(n_init):
            if evals >= budget:
                break
            x = lb + (ub - lb) * rng.rand(dim)
            f = func(x)
            evals += 1
            init_points.append(x)
            init_vals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # Fill remaining vertices if necessary
        while len(init_points) < n_vertices and evals < budget:
            x = lb + (ub - lb) * rng.rand(dim)
            f = func(x)
            evals += 1
            init_points.append(x)
            init_vals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        simplex = np.array(init_points[:n_vertices])
        f_simplex = np.array(init_vals[:n_vertices])

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_no_improve = max(10, dim * 2)
        no_improve_count = 0
        step_scale = 0.1 * np.mean(ub - lb)  # initial step size for restart

        while evals < budget:
            # Order vertices by fitness
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evals += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
                report_best(best_f, best_x)

            improved = False
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
                improved = True
            elif fr < f_simplex[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lb, ub)
                fe = func(xe)
                evals += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                    report_best(best_f, best_x)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                improved = True
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evals += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                    report_best(best_f, best_x)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                    improved = True
                else:
                    # Shrink
                    for i in range(1, n_vertices):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lb, ub)
                        fi = func(simplex[i])
                        evals += 1
                        f_simplex[i] = fi
                        if fi < best_f:
                            best_f = fi
                            best_x = simplex[i].copy()
                            report_best(best_f, best_x)

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Restart on stagnation
            if no_improve_count >= max_no_improve and evals < budget:
                n_trials = min(n_vertices - 1, budget - evals)
                for _ in range(n_trials):
                    perturbation = step_scale * rng.randn(dim)
                    x_new = np.clip(best_x + perturbation, lb, ub)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                    # Replace worst vertex if better
                    if f_new < f_simplex[-1]:
                        simplex[-1] = x_new
                        f_simplex[-1] = f_new
                # Adapt step scale
                if best_f == f_simplex[0]:  # no improvement in restart
                    step_scale *= 0.5
                else:
                    step_scale *= 1.2
                no_improve_count = 0

            if evals >= budget:
                break

        return best_f, best_x