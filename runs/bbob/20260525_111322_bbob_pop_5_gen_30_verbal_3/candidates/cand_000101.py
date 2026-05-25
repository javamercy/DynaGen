import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        rng = self.rng
        budget = self.budget

        n = dim
        n_vertices = n + 1

        # Initialize simplex uniformly random
        simplex = rng.uniform(lb, ub, size=(n_vertices, dim))
        f_simplex = np.full(n_vertices, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        # Evaluate initial simplex
        for i in range(n_vertices):
            if evals >= budget:
                break
            x = simplex[i]
            f = func(x)
            evals += 1
            f_simplex[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Nelder-Mead parameters (adaptive sigma)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_no_improve = max(10, dim * 2)
        no_improve_count = 0

        while evals < budget:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evals += 1
            if evals >= budget:
                break
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
                report_best(best_f, best_x)

            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
                no_improve_count = 0
            elif fr < f_simplex[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lb, ub)
                fe = func(xe)
                evals += 1
                if evals >= budget:
                    break
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
                no_improve_count = 0
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evals += 1
                if evals >= budget:
                    break
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                    report_best(best_f, best_x)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                    no_improve_count = 0
                else:
                    # Shrink with adaptive sigma
                    sigma_adapt = sigma * (1 + 0.1 * no_improve_count)  # increase sigma when stuck
                    for i in range(1, n_vertices):
                        simplex[i] = simplex[0] + sigma_adapt * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lb, ub)
                        fi = func(simplex[i])
                        evals += 1
                        if evals >= budget:
                            break
                        f_simplex[i] = fi
                        if fi < best_f:
                            best_f = fi
                            best_x = simplex[i].copy()
                            report_best(best_f, best_x)
                    no_improve_count += 1

            # Restart if no improvement
            if no_improve_count >= max_no_improve and evals < budget:
                # Keep best point
                simplex[0] = best_x
                f_simplex[0] = best_f
                # Generate new points by perturbing best with random step
                step_size = 0.1 * (ub - lb)  # scale relative to bounds
                for i in range(1, n_vertices):
                    sample = best_x + step_size * rng.normal(0, 1, dim)
                    sample = np.clip(sample, lb, ub)
                    fi = func(sample)
                    evals += 1
                    if evals >= budget:
                        break
                    f_simplex[i] = fi
                    if fi < best_f:
                        best_f = fi
                        best_x = sample.copy()
                        report_best(best_f, best_x)
                # Reset parameters
                no_improve_count = 0
                sigma = 0.5

        return best_f, best_x