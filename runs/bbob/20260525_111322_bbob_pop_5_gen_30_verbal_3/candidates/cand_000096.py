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

        # Default NM parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Adaptive parameter bounds
        alpha_min, alpha_max = 0.5, 2.0
        gamma_min, gamma_max = 1.0, 4.0
        rho_min, rho_max = 0.1, 0.9
        sigma_min, sigma_max = 0.1, 0.9

        max_no_improve = max(10, dim * 2)
        no_improve_count = 0

        while evals < budget:
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = f_simplex[-1]
            second_worst = f_simplex[-2]
            best_f_current = f_simplex[0]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evals += 1
            if evals >= budget:
                break
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
                report_best(best_f, best_x)

            improved = False
            if fr < f_simplex[0]:  # better than best, try expansion
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
                improved = True
                # Increase alpha, gamma; decrease rho
                alpha = min(alpha * 1.1, alpha_max)
                gamma = min(gamma * 1.1, gamma_max)
                rho = max(rho * 0.9, rho_min)
            elif fr < second_worst:  # better than second worst, accept reflection
                simplex[-1] = xr
                f_simplex[-1] = fr
                improved = True
                # Increase alpha, gamma; decrease rho
                alpha = min(alpha * 1.05, alpha_max)
                gamma = min(gamma * 1.05, gamma_max)
                rho = max(rho * 0.95, rho_min)
            else:  # contraction
                if fr < f_worst:
                    xc = centroid + rho * (xr - centroid)  # outside contraction?
                else:
                    xc = centroid - rho * (centroid - worst)
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evals += 1
                if evals >= budget:
                    break
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                    report_best(best_f, best_x)
                if fc < min(fr, f_worst):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                    improved = True
                    # Slight increase alpha, gamma; decrease rho
                    alpha = min(alpha * 1.02, alpha_max)
                    gamma = min(gamma * 1.02, gamma_max)
                    rho = max(rho * 0.98, rho_min)
                else:  # shrink
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
                    improved = False
                    # Decrease alpha, gamma; increase rho, sigma
                    alpha = max(alpha * 0.9, alpha_min)
                    gamma = max(gamma * 0.9, gamma_min)
                    rho = min(rho * 1.1, rho_max)
                    sigma = min(sigma * 1.1, sigma_max)

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Restart if stagnation
            if no_improve_count >= max_no_improve and evals < budget:
                # Reset parameters
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                # Keep best vertex
                simplex[0] = best_x
                f_simplex[0] = best_f
                for i in range(1, n_vertices):
                    simplex[i] = rng.uniform(lb, ub, size=dim)
                    fi = func(simplex[i])
                    evals += 1
                    if evals >= budget:
                        break
                    f_simplex[i] = fi
                    if fi < best_f:
                        best_f = fi
                        best_x = simplex[i].copy()
                        report_best(best_f, best_x)
                no_improve_count = 0

        return best_f, best_x