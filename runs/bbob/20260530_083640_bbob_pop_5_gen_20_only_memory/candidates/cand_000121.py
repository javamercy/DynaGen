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

        # Reserve for DE and local search
        # Exploitation: allocate 60% of budget to local search, but at least dim+1 for simplex
        local_budget_fraction = 0.6
        min_local = max(dim + 1, 10)
        max_de_budget = max(budget - min_local, int(budget * (1 - local_budget_fraction)))
        de_budget = max_de_budget
        # Ensure de_budget is positive and leaves enough for local
        if de_budget < 10:
            de_budget = max(10, budget - min_local)
        local_budget = budget - de_budget

        # Initial population size: small for exploitation
        pop_size = max(3 * dim, 5)
        pop_size = min(pop_size, de_budget // 2)  # at least a few generations
        pop_size = max(dim + 1, pop_size)
        if pop_size > de_budget:
            pop_size = de_budget

        # Latin Hypercube initial pop
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial pop
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
            if evals >= de_budget:
                break
        if evals >= budget:
            return best_val, best_x

        # DE with current-to-best mutation (exploitative)
        # Budget for DE: allow remaining evaluations up to de_budget
        de_remaining = de_budget - evals
        if de_remaining > 0:
            # Number of generations: at least 1, at most de_remaining // pop_size
            max_gens = max(1, de_remaining // pop_size)
            for gen in range(max_gens):
                if evals >= de_budget:
                    break
                progress = gen / max_gens
                F = 0.6 - 0.4 * progress  # from 0.6 to 0.2
                CR = 0.8 - 0.3 * progress  # from 0.8 to 0.5
                for i in range(pop_size):
                    if evals >= de_budget:
                        break
                    # Mutation: current-to-best/1
                    # Select two random distinct indices different from i and best index
                    best_idx = np.argmin(pop_fitness)
                    candidates = [j for j in range(pop_size) if j != i and j != best_idx]
                    rng.shuffle(candidates)
                    a, b = candidates[:2]
                    mutant = pop[i] + F * (pop[best_idx] - pop[i]) + F * (pop[a] - pop[b])
                    # Crossover
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
        # Local search: Nelder-Mead with aggressive expansion
        remaining = budget - evals
        if remaining > 0:
            # Create simplex around best
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.1 * (ub[i] - lb[i])  # larger initial step
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

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.5  # more expansion
            rho = 0.5
            sigma = 0.5
            max_iter = 200
            for _ in range(max_iter):
                if evals >= budget:
                    break
                # Sort
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