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
        simplex = rng.uniform(lb, ub, (n_vertices, dim))
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

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_no_improve = max(10, dim * 2)
        no_improve_count = 0
        F = 0.8
        p_mutate = 0.3

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

            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
                no_improve_count = 0
            elif fr < f_simplex[0]:
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
                    for i in range(1, n_vertices):
                        if evals >= budget:
                            break
                        if rng.rand() < p_mutate:
                            idx_pool = list(range(1, n_vertices))
                            idx_pool.remove(i)
                            r1, r2 = rng.choice(idx_pool, 2, replace=False)
                            mutant = simplex[0] + F * (simplex[r1] - simplex[r2])
                            mutant = np.clip(mutant, lb, ub)
                        else:
                            mutant = simplex[0] + sigma * (simplex[i] - simplex[0])
                            mutant = np.clip(mutant, lb, ub)
                        fi = func(mutant)
                        evals += 1
                        if fi < best_f:
                            best_f = fi
                            best_x = mutant.copy()
                            report_best(best_f, best_x)
                        f_simplex[i] = fi
                    no_improve_count += 1

            if no_improve_count >= max_no_improve and evals < budget:
                simplex[0] = best_x
                f_simplex[0] = best_f
                for i in range(1, n_vertices):
                    if evals >= budget:
                        break
                    simplex[i] = rng.uniform(lb, ub, size=dim)
                    fi = func(simplex[i])
                    evals += 1
                    if fi < best_f:
                        best_f = fi
                        best_x = simplex[i].copy()
                        report_best(best_f, best_x)
                    f_simplex[i] = fi
                no_improve_count = 0

        return best_f, best_x