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

        best_val = np.inf
        best_x = None
        evals = 0

        # Small initial Latin Hypercube
        init_pop_size = min(2 * dim, budget // 10)
        if init_pop_size < 2:
            init_pop_size = 2
        if init_pop_size > budget:
            init_pop_size = budget
        lhs = self._latin_hypercube(init_pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs
        for i in range(init_pop_size):
            if evals >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        if best_x is None:
            # fallback
            x = (lb + ub) / 2
            val = func(x)
            evals += 1
            best_val = val
            best_x = x.copy()
            report_best(best_val, best_x)

        # Short DE with small population
        de_pop_size = min(4 * dim, budget // 5)
        if de_pop_size < 2:
            de_pop_size = 2
        if evals + de_pop_size * (dim + 1) >= budget:
            de_pop_size = max(2, budget - evals - dim - 1)
        if de_pop_size > 0:
            lhs2 = self._latin_hypercube(de_pop_size, dim, rng)
            de_pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs2
            de_fitness = np.full(de_pop_size, np.inf)
            for i in range(de_pop_size):
                if evals >= budget:
                    break
                x = np.clip(de_pop[i], lb, ub)
                val = func(x)
                evals += 1
                de_fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

            F = 0.5
            CR = 0.9
            max_de_iter = min(10, (budget - evals) // de_pop_size)
            for _ in range(max_de_iter):
                if evals >= budget:
                    break
                for i in range(de_pop_size):
                    if evals >= budget:
                        break
                    indices = [j for j in range(de_pop_size) if j != i]
                    rng.shuffle(indices)
                    a, b, c = indices[:3]
                    mutant = de_pop[a] + F * (de_pop[b] - de_pop[c])
                    j_rand = rng.randint(dim)
                    trial = np.where(rng.rand(dim) < CR, mutant, de_pop[i])
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < de_fitness[i]:
                        de_pop[i] = trial
                        de_fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)

        # Intensified local search via Nelder-Mead with restarts
        while evals < budget:
            # Build simplex around best
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.05 * (ub[i] - lb[i])
                x = best_x.copy()
                x[i] = min(ub[i], max(lb[i], x[i] + step))
                simplex[i + 1] = x
            simplex_vals = np.full(dim + 1, np.inf)
            simplex_vals[0] = best_val
            for i in range(1, dim + 1):
                if evals >= budget:
                    break
                val = func(simplex[i])
                evals += 1
                simplex_vals[i] = val
                if val < best_val:
                    best_val = val
                    best_x = simplex[i].copy()
                    report_best(best_val, best_x)

            if evals >= budget:
                break

            # Nelder-Mead parameters
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = min(50, (budget - evals) // (dim + 1) + 1)
            for _ in range(max_iter):
                if evals >= budget:
                    break
                order = np.argsort(simplex_vals)
                simplex = simplex[order]
                simplex_vals = simplex_vals[order]
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                yr = func(xr)
                evals += 1
                if yr < simplex_vals[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    ye = func(xe)
                    evals += 1
                    if ye < yr:
                        simplex[-1] = xe
                        simplex_vals[-1] = ye
                    else:
                        simplex[-1] = xr
                        simplex_vals[-1] = yr
                elif yr < simplex_vals[-2]:
                    simplex[-1] = xr
                    simplex_vals[-1] = yr
                else:
                    if yr < simplex_vals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    yc = func(xc)
                    evals += 1
                    if yc < simplex_vals[-1]:
                        simplex[-1] = xc
                        simplex_vals[-1] = yc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if evals >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            y = func(simplex[i])
                            evals += 1
                            simplex_vals[i] = y
                            if y < best_val:
                                best_val = y
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)

                current_best_idx = np.argmin(simplex_vals)
                if simplex_vals[current_best_idx] < best_val:
                    best_val = simplex_vals[current_best_idx]
                    best_x = simplex[current_best_idx].copy()
                    report_best(best_val, best_x)

                if evals >= budget:
                    break

            # Restart: if converged, add random perturbation to best and rebuild
            # Check if simplex is too small (std across vertices)
            span = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.mean(span / (ub - lb)) < 1e-4:
                # Perturb best
                x = best_x + 0.1 * (ub - lb) * rng.randn(dim)
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                # Try another random point
                if evals < budget:
                    x2 = best_x + 0.1 * (ub - lb) * rng.randn(dim)
                    x2 = np.clip(x2, lb, ub)
                    val2 = func(x2)
                    evals += 1
                    if val2 < best_val:
                        best_val = val2
                        best_x = x2.copy()
                        report_best(best_val, best_x)

        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs