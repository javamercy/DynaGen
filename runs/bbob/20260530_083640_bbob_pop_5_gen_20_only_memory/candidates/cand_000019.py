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
        pop_size = max(pop_size, 2 * dim)  # ensure enough for mutation
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

        # Reserve for local search
        reserve = max(2 * dim, 20)
        reserve = min(reserve, budget - evals)  # cannot exceed remaining
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        # Adaptive DE parameters
        for gen in range(max_gen):
            # Schedule F and CR
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac  # 0.9 -> 0.5
            CR = 0.5 + 0.4 * frac  # 0.5 -> 0.9
            for i in range(pop_size):
                # Mutation
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                # Evaluation
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

        # Local search: Nelder-Mead simplex around best
        remaining = budget - evals
        if remaining > 0:
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.1 * (ub[i] - lb[i])  # larger step than parent
                x = best_x.copy()
                x[i] = min(ub[i], max(lb[i], x[i] + step))
                simplex[i + 1] = x

            simplex_vals = np.full(dim + 1, np.inf)
            simplex_vals[0] = best_val  # reuse best_val, no new eval
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

            # Nelder-Mead parameters
            alpha = 1.1
            gamma = 2.1
            rho = 0.4
            sigma = 0.6
            max_iter = 100
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
                    # Expansion
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
                    # Contraction
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

                # Update best
                current_best_idx = np.argmin(simplex_vals)
                if simplex_vals[current_best_idx] < best_val:
                    best_val = simplex_vals[current_best_idx]
                    best_x = simplex[current_best_idx].copy()
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