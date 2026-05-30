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

        # Evaluate at least one feasible point
        best_x = np.clip(rng.uniform(lb, ub), lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # Fallback for very small budgets
        if budget <= 2*dim + 10:
            for _ in range(budget - evals):
                x = np.clip(rng.uniform(lb, ub), lb, ub)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Reserve evaluations for local search (Nelder-Mead requires dim+1 points)
        reserve_local = min(max(10, dim + 1), budget // 4)
        # Initial population size: try to allocate a decent number, but leave room for DE and local
        init_pop_size = max(2*dim, 5 * dim)
        init_pop_size = min(init_pop_size, (budget - reserve_local) // 2)
        if init_pop_size < 2:
            init_pop_size = 2
        # Ensure we have enough budget for at least one generation
        if init_pop_size * 2 > budget - reserve_local:
            init_pop_size = max(2, (budget - reserve_local) // 2)

        # Latin Hypercube initial population
        def lhs(n, d):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    samples[i, j] = (perm[i] + rng.uniform(0, 1)) / n
            return samples

        pop = lb + (ub - lb) * lhs(init_pop_size, dim)
        pop_fitness = np.full(init_pop_size, np.inf)

        for i in range(init_pop_size):
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

        # Differential Evolution with adaptive parameters
        remaining_for_de = budget - evals - reserve_local
        if remaining_for_de > 0:
            max_gens = min(50, remaining_for_de // init_pop_size)
            for gen in range(max_gens):
                if evals >= budget - reserve_local:
                    break
                progress = gen / max_gens
                F = 0.8 - 0.6 * progress
                CR = 0.9 - 0.4 * progress

                for i in range(init_pop_size):
                    if evals >= budget - reserve_local:
                        break
                    idxs = [j for j in range(init_pop_size) if j != i]
                    rng.shuffle(idxs)
                    a, b, c = idxs[:3]
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
        while evals < budget:
            remaining = budget - evals
            simplex_size = min(dim + 1, remaining)
            simplex = np.zeros((simplex_size, dim))
            simplex[0] = best_x.copy()
            for i in range(1, simplex_size):
                step = 0.05 * (ub[i-1] - lb[i-1]) if i-1 < dim else 0.05 * (ub[0] - lb[0])
                if step == 0:
                    step = 0.01
                x = best_x.copy()
                idx = (i-1) % dim
                x[idx] = min(ub[idx], max(lb[idx], x[idx] + step))
                simplex[i] = x

            simplex_vals = np.full(simplex_size, np.inf)
            for i in range(simplex_size):
                if evals >= budget:
                    break
                val = func(simplex[i])
                evals += 1
                simplex_vals[i] = val
                if val < best_val:
                    best_val = val
                    best_x = simplex[i].copy()
                    report_best(best_val, best_x)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = 100
            for _ in range(max_iter):
                if evals >= budget:
                    break
                order = np.argsort(simplex_vals)
                simplex = simplex[order]
                simplex_vals = simplex_vals[order]

                centroid = np.mean(simplex[:-1], axis=0)
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
                        for i in range(1, simplex_size):
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

                best_idx = np.argmin(simplex_vals)
                if simplex_vals[best_idx] < best_val:
                    best_val = simplex_vals[best_idx]
                    best_x = simplex[best_idx].copy()
                    report_best(best_val, best_x)

                if evals >= budget:
                    break

        return best_val, best_x