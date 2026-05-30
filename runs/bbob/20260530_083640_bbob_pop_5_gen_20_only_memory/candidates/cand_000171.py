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

        # Population size
        pop_size = min(budget // 2, max(3 * dim, min(15 + int(dim**0.5), budget // 2)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget)

        # Latin Hypercube initialization
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

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

        # Adaptive DE
        reserve = max(2 * dim, 20)
        reserve = min(reserve, budget - evals)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

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

        # Nelder-Mead refinement
        remaining = budget - evals
        if remaining > 0:
            x0 = best_x.copy()
            f0 = best_val
            # Build initial simplex
            simplex_pts = [x0.copy()]
            for i in range(dim):
                x = x0.copy()
                delta = 0.05 * (ub - lb)
                x[i] = min(ub[i], max(lb[i], x[i] + delta[i]))
                simplex_pts.append(x)
            simplex_vals = []
            for x in simplex_pts:
                if evals >= budget:
                    break
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                simplex_vals.append(val)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            # Nelder-Mead iterations
            while evals < budget:
                # Sort simplex
                order = np.argsort(simplex_vals)
                simplex_pts = [simplex_pts[i] for i in order]
                simplex_vals = [simplex_vals[i] for i in order]
                # Centroid of best n points
                centroid = np.mean(simplex_pts[:-1], axis=0)
                # Reflection
                xr = centroid + (centroid - simplex_pts[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals += 1
                if fr < simplex_vals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex_pts[-1])
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex_pts[-1] = xe
                        simplex_vals[-1] = fe
                    else:
                        simplex_pts[-1] = xr
                        simplex_vals[-1] = fr
                elif fr < simplex_vals[-2]:
                    simplex_pts[-1] = xr
                    simplex_vals[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_vals[-1]:
                        xc = centroid + 0.5 * (xr - centroid)
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        evals += 1
                        if fc < fr:
                            simplex_pts[-1] = xc
                            simplex_vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex_pts)):
                                simplex_pts[i] = simplex_pts[0] + 0.5 * (simplex_pts[i] - simplex_pts[0])
                                simplex_pts[i] = np.clip(simplex_pts[i], lb, ub)
                                if evals >= budget:
                                    break
                                val = func(simplex_pts[i])
                                evals += 1
                                simplex_vals[i] = val
                    else:
                        xc = centroid - 0.5 * (centroid - simplex_pts[-1])
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        evals += 1
                        if fc < simplex_vals[-1]:
                            simplex_pts[-1] = xc
                            simplex_vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex_pts)):
                                simplex_pts[i] = simplex_pts[0] + 0.5 * (simplex_pts[i] - simplex_pts[0])
                                simplex_pts[i] = np.clip(simplex_pts[i], lb, ub)
                                if evals >= budget:
                                    break
                                val = func(simplex_pts[i])
                                evals += 1
                                simplex_vals[i] = val
                # Update best
                for i, f in enumerate(simplex_vals):
                    if f < best_val:
                        best_val = f
                        best_x = simplex_pts[i].copy()
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