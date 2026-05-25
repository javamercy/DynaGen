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

        n = dim
        n_vertices = n + 1
        # Initialize simplex uniformly random
        simplex = rng.uniform(lb, ub, size=(n_vertices, dim))
        f_simplex = np.full(n_vertices, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

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

        # Nelder-Mead parameters
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
                    # Shrink
                    for i in range(1, n_vertices):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
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

            # Restart condition
            if no_improve_count >= max_no_improve and evals < budget:
                # Keep best point
                simplex[0] = best_x
                f_simplex[0] = best_f
                # Compute covariance from current simplex (excluding best)
                points = simplex[1:]  # shape (n_vertices-1, dim)
                # Add small regularization to avoid singularity
                cov = np.cov(points, rowvar=False) + 1e-10 * np.eye(dim)
                # Generate new points from multivariate normal around best
                for i in range(1, n_vertices):
                    sample = rng.multivariate_normal(best_x, cov, size=1)[0]
                    sample = np.clip(sample, lb, ub)
                    simplex[i] = sample
                    fi = func(sample)
                    evals += 1
                    if evals >= budget:
                        break
                    f_simplex[i] = fi
                    if fi < best_f:
                        best_f = fi
                        best_x = sample.copy()
                        report_best(best_f, best_x)
                # DE/current-to-rand/1 mutation (rotation-invariant) on the new simplex
                if evals < budget:
                    r1, r2, r3 = rng.choice(n_vertices, size=3, replace=False)
                    # current-to-rand/1: xi + K*(xr1 - xi) + F*(xr2 - xr3)
                    K = rng.uniform(0.5, 1.0)
                    F = rng.uniform(0.5, 1.0)
                    # Choose random target index
                    target_idx = rng.randint(n_vertices)
                    trial = simplex[target_idx] + K * (simplex[r1] - simplex[target_idx]) + F * (simplex[r2] - simplex[r3])
                    trial = np.clip(trial, lb, ub)
                    ftrial = func(trial)
                    evals += 1
                    if ftrial < best_f:
                        best_f = ftrial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                    if ftrial < f_simplex[-1]:
                        simplex[-1] = trial
                        f_simplex[-1] = ftrial
                no_improve_count = 0

        return best_f, best_x