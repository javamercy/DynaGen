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

        # Population size for DE
        pop_size = min(budget // 2, max(3 * dim, min(15 + int(dim**0.5), budget // 2)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget)

        # Latin Hypercube initial population
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial points
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Reserve for local search (at least dim+1 evaluations for simplex initialization)
        reserve = min(budget - evals, max(10 * dim, 50))
        reserve = max(reserve, dim + 1)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        # Adaptive DE
        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                if evals >= budget:
                    return best_val, best_x

        # Nelder-Mead local search
        remaining = budget - evals
        if remaining >= dim + 1:
            # Initialize simplex around best_x
            step = 0.1 * (ub - lb)
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                point = best_x.copy()
                point[i] = np.clip(point[i] + step[i], lb[i], ub[i])
                simplex[i + 1] = point

            f_simplex = np.full(dim + 1, np.inf)
            for i in range(dim + 1):
                if evals >= budget:
                    break
                f_simplex[i] = func(simplex[i])
                evals += 1
                if f_simplex[i] < best_val:
                    best_val = f_simplex[i]
                    best_x = simplex[i].copy()
                    report_best(best_val, best_x)

            # Standard Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            while evals < budget:
                # Order simplex by fitness
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
                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                        if fe < best_val:
                            best_val = fe
                            best_x = xe.copy()
                            report_best(best_val, best_x)
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                        if fr < best_val:
                            best_val = fr
                            best_x = xr.copy()
                            report_best(best_val, best_x)
                elif fr < f_simplex[-2]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                    if fr < best_val:
                        best_val = fr
                        best_x = xr.copy()
                        report_best(best_val, best_x)
                else:
                    # Contraction
                    if fr < f_simplex[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    evals += 1
                    if fc < min(f_simplex[-1], fr):
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                        if fc < best_val:
                            best_val = fc
                            best_x = xc.copy()
                            report_best(best_val, best_x)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            f_simplex[i] = func(simplex[i])
                            evals += 1
                            if f_simplex[i] < best_val:
                                best_val = f_simplex[i]
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)
                            if evals >= budget:
                                break
        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs