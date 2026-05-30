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

        # Reserve for local search
        reserve = max(2 * dim, 20)
        reserve = min(reserve, budget - 1)  # at least 1 eval for initial

        # Population size for DE: at least 2*dim, at most budget/2
        pop_size = max(2 * dim, min(40, budget // 4))
        pop_size = min(pop_size, budget - reserve)
        if pop_size < 4:
            pop_size = 4

        # Latin Hypercube initialization
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs
        pop = np.clip(pop, lb, ub)

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = pop[i]
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # DE generations
        de_budget = budget - evals - reserve
        if de_budget > 0:
            max_gen = max(1, de_budget // pop_size)
            # Adaptive parameters for each individual
            F_vals = np.full(pop_size, 0.7)
            CR_vals = np.full(pop_size, 0.5)
            # Probability of using rand/1 (vs current-to-best/1)
            p_rand = 0.5

            for gen in range(max_gen):
                if evals >= budget - reserve:
                    break
                # Update probability: decrease over time for more exploitation
                p_rand = max(0.2, 0.5 - 0.3 * gen / max_gen)

                for i in range(pop_size):
                    if evals >= budget - reserve:
                        break
                    # Generate F and CR for this individual (self-adaptive)
                    F = F_vals[i] + 0.1 * rng.randn()
                    F = np.clip(F, 0.1, 0.9)
                    CR = CR_vals[i] + 0.1 * rng.randn()
                    CR = np.clip(CR, 0.0, 1.0)

                    # Mutation strategy choice
                    if rng.rand() < p_rand:
                        # DE/rand/1
                        indices = [j for j in range(pop_size) if j != i]
                        rng.shuffle(indices)
                        a, b, c = indices[:3]
                        mutant = pop[a] + F * (pop[b] - pop[c])
                    else:
                        # DE/current-to-best/1
                        indices = [j for j in range(pop_size) if j != i]
                        rng.shuffle(indices)
                        a, b = indices[:2]
                        mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])

                    # Crossover
                    j_rand = rng.randint(dim)
                    trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                    trial = np.clip(trial, lb, ub)

                    val = func(trial)
                    evals += 1
                    if val < pop_fitness[i]:
                        pop[i] = trial
                        pop_fitness[i] = val
                        # Update adaptive parameters on success
                        F_vals[i] = F
                        CR_vals[i] = CR
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                    if evals >= budget:
                        return best_val, best_x

        # Nelder-Mead local search
        remaining = budget - evals
        if remaining > 0:
            # Build simplex around best
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.1 * (ub[i] - lb[i])
                if step == 0:
                    step = 0.01
                x = best_x.copy()
                x[i] = min(ub[i], max(lb[i], x[i] + step))
                simplex[i + 1] = x

            simplex_vals = np.full(dim + 1, np.inf)
            simplex_vals[0] = best_val  # no new eval
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

            # Nelder-Mead coefficients
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = 100
            # Adaptive simplex size: shrink as evaluations decrease
            initial_size = np.max(simplex.max(axis=0) - simplex.min(axis=0))

            for it in range(max_iter):
                if evals >= budget:
                    break
                # Sort simplex
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

                # Optional break if simplex is very small and budget low
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