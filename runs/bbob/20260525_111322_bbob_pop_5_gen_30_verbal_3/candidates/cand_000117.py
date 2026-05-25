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

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_no_improve = max(10, dim * 2)
        no_improve_count = 0
        F = 0.5
        Cr = 0.9

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

            if no_improve_count >= max_no_improve and evals < budget:
                # Anisotropic restart: estimate covariance from current simplex (excluding best)
                simplex_copy = simplex[1:]  # exclude best
                mean_vec = best_x.copy()
                cov_mat = np.cov(simplex_copy.T) + 1e-8 * np.eye(dim)
                # Sample new vertices from multivariate normal
                simplex[0] = best_x
                f_simplex[0] = best_f
                for i in range(1, n_vertices):
                    # sample until within bounds
                    while True:
                        candidate = rng.multivariate_normal(mean_vec, cov_mat)
                        if np.all(candidate >= lb) and np.all(candidate <= ub):
                            break
                    simplex[i] = candidate
                    fi = func(simplex[i])
                    evals += 1
                    if evals >= budget:
                        break
                    f_simplex[i] = fi
                    if fi < best_f:
                        best_f = fi
                        best_x = simplex[i].copy()
                        report_best(best_f, best_x)
                # DE/rand/1 mutation after restart
                if evals < budget:
                    indices = list(range(n_vertices))
                    rng.shuffle(indices)
                    r0, r1, r2 = indices[:3]
                    mutant = simplex[r0] + F * (simplex[r1] - simplex[r2])
                    mutant = np.clip(mutant, lb, ub)
                    target_idx = rng.randint(n_vertices)
                    trial = simplex[target_idx].copy()
                    for j in range(dim):
                        if rng.rand() < Cr or j == rng.randint(dim):
                            trial[j] = mutant[j]
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
```