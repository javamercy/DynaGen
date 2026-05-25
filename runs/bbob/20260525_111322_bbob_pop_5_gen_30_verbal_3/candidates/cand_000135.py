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
        scale = np.mean(ub - lb)

        best_x = None
        best_f = np.inf
        evals = 0

        # Initial random sampling
        n_init = min(dim + 1, max(1, int(0.1 * budget)))
        for _ in range(n_init):
            if evals >= budget:
                break
            x = lb + (ub - lb) * rng.rand(dim)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        if evals >= budget:
            return best_f, best_x

        # Initialize simplex
        n_vertices = dim + 1
        simplex = np.zeros((n_vertices, dim))
        f_simplex = np.full(n_vertices, np.inf)
        simplex[0] = best_x.copy()
        f_simplex[0] = best_f
        for i in range(1, n_vertices):
            if evals >= budget:
                break
            x = lb + (ub - lb) * rng.rand(dim)
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

        # Pattern search directions
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        # For anisotropic sampling: maintain covariance estimate of simplex (excluding worst)
        # We'll update it at each restart
        def compute_cov(points):
            # points: (n, dim)
            if len(points) < 2:
                return np.eye(dim) * (scale ** 2)
            return np.cov(points, rowvar=False) + np.eye(dim) * 1e-12

        while evals < budget:
            # Order
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
                    improved = True
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

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= max_no_improve and evals < budget:
                # Local pattern search
                step = 0.1 * scale
                pattern_evals = min(50, budget - evals)
                pattern_used = 0
                x_current = best_x.copy()
                f_current = best_f
                while pattern_used < pattern_evals:
                    found = False
                    for d in directions:
                        if pattern_used >= pattern_evals:
                            break
                        candidate = np.clip(x_current + step * d, lb, ub)
                        val = func(candidate)
                        pattern_used += 1
                        evals += 1
                        if val < best_f:
                            best_f = val
                            best_x = candidate.copy()
                            report_best(best_f, best_x)
                            x_current = candidate
                            f_current = val
                            found = True
                            step *= 1.2
                            break
                    if not found:
                        step *= 0.5
                        if step < 1e-12 * scale:
                            break

                # Anisotropic restart using DE/rand/1 mutation
                # Estimate covariance from the best (n_vertices-1) points (or from simplex if many visited)
                # Use current simplex points (excluding worst) to compute covariance
                best_indices = np.argsort(f_simplex)[:n_vertices//2]  # use top half
                if len(best_indices) < 2:
                    cov = np.eye(dim) * (scale ** 2)
                else:
                    cov = compute_cov(simplex[best_indices])
                # Generate new population from multivariate normal centered at best_x
                pop_size = n_vertices - 1
                # Use Cholesky decomposition to sample
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    L = np.eye(dim) * (scale ** 2) ** 0.5
                pop = best_x + L @ rng.randn(dim, pop_size).T
                pop = np.clip(pop, lb, ub)
                
                # Build new simplex: keep best_x as first vertex
                new_simplex = np.zeros((n_vertices, dim))
                new_f = np.full(n_vertices, np.inf)
                new_simplex[0] = best_x.copy()
                new_f[0] = best_f
                for i in range(1, n_vertices):
                    if evals >= budget:
                        break
                    # Differential mutation: base + F * (r1 - r2)
                    idx = rng.choice(pop_size, size=3, replace=False)
                    F = 0.5 + 0.5 * rng.rand()
                    mutant = pop[idx[0]] + F * (pop[idx[1]] - pop[idx[2]])
                    mutant = np.clip(mutant, lb, ub)
                    # Crossover with best_x (binary)
                    cr = 0.9
                    mask = rng.rand(dim) < cr
                    if not np.any(mask):
                        mask[rng.randint(dim)] = True
                    trial = np.where(mask, mutant, new_simplex[0])
                    trial = np.clip(trial, lb, ub)
                    ft = func(trial)
                    evals += 1
                    new_simplex[i] = trial
                    new_f[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                # Also add one purely random point from the anisotropic distribution
                if evals < budget:
                    x_rand = best_x + L @ rng.randn(dim)
                    x_rand = np.clip(x_rand, lb, ub)
                    f_rand = func(x_rand)
                    evals += 1
                    worst_idx = np.argmax(new_f)
                    if f_rand < new_f[worst_idx]:
                        new_simplex[worst_idx] = x_rand
                        new_f[worst_idx] = f_rand
                    if f_rand < best_f:
                        best_f = f_rand
                        best_x = x_rand.copy()
                        report_best(best_f, best_x)
                simplex = new_simplex
                f_simplex = new_f
                no_improve_count = 0

            if evals >= budget:
                break

        return best_f, best_x