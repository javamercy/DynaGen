import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
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
        simplex = np.zeros((n_vertices, dim))
        for i in range(n_vertices):
            simplex[i] = rng.uniform(lb, ub, size=dim)
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

        alpha = 1.5
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        alpha_min, alpha_max = 1.0, 2.0
        gamma_min, gamma_max = 1.5, 3.0
        rho_min, rho_max = 0.25, 0.75
        sigma_min, sigma_max = 0.25, 0.75

        max_no_improve = max(5, 2 * dim)
        no_improve_count = 0
        success_streak = 0

        while evals < budget:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            centroid = np.mean(simplex[:-1], axis=0)

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

            improved = False
            if fr < f_simplex[-2]:
                if fr < f_simplex[0]:
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
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                    improved = True
            else:
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
                    improved = True
                else:
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
                    alpha = max(alpha_min, alpha * 0.95)
                    gamma = max(gamma_min, gamma * 0.95)
                    rho = max(rho_min, rho * 0.95)
                    sigma = max(sigma_min, sigma * 0.95)
                    success_streak = 0
                    no_improve_count += 1
                    continue

            if improved:
                success_streak += 1
                if success_streak > 5:
                    alpha = min(alpha_max, alpha * 1.05)
                    gamma = min(gamma_max, gamma * 1.05)
                    rho = min(rho_max, rho * 1.05)
                    sigma = min(sigma_max, sigma * 1.05)
                    success_streak = 0
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= max_no_improve and evals < budget:
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
                alpha = 1.5
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                success_streak = 0

        return best_f, best_x