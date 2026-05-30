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

        # population size
        pop_size = max(4, min(10, budget // 10))
        de_budget = max(2 * (pop_size + 1), budget // 3)
        de_evals = 0

        # Latin hypercube sampling for initial population
        intervals = np.linspace(0, 1, pop_size + 1)
        points = np.zeros((pop_size, dim))
        for d in range(dim):
            samples = rng.uniform(intervals[:-1], intervals[1:], size=pop_size)
            points[:, d] = samples[rng.permutation(pop_size)]
        pop = lb + points * (ub - lb)
        pop_fit = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None

        for i in range(pop_size):
            val = func(pop[i])
            de_evals += 1
            pop_fit[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        max_gen = (de_budget - pop_size) // pop_size
        for gen in range(max_gen):
            # Adaptive mutation factor and crossover rate
            F = 0.9 * (1 - 0.5 * gen / max_gen) if max_gen > 0 else 0.5
            CR = 0.9 * (1 - 0.3 * gen / max_gen) if max_gen > 0 else 0.5
            for i in range(pop_size):
                if de_evals >= de_budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                de_evals += 1
                if val < pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            if de_evals >= de_budget:
                break

        # Phase 2: Nelder-Mead local search with adaptive simplex size
        remaining = budget - de_evals
        if remaining < dim + 2:
            for _ in range(remaining):
                x = best_x + 0.01 * (ub - lb) * rng.randn(dim)
                x = np.clip(x, lb, ub)
                val = func(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
        else:
            while remaining > 0:
                step_factor = 0.1 * (1 - (remaining / budget)) + 0.001
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = best_x.copy()
                for i in range(dim):
                    step = step_factor * (ub[i] - lb[i])
                    x = best_x.copy()
                    x[i] = min(ub[i], max(lb[i], x[i] + step))
                    simplex[i+1] = x

                simplex_vals = np.full(dim + 1, np.inf)
                simplex_vals[0] = best_val
                for i in range(1, dim + 1):
                    if remaining <= 0:
                        break
                    val = func(simplex[i])
                    remaining -= 1
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
                    if remaining <= 0:
                        break
                    order = np.argsort(simplex_vals)
                    simplex = simplex[order]
                    simplex_vals = simplex_vals[order]
                    centroid = np.mean(simplex[:-1], axis=0)

                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    yr = func(xr)
                    remaining -= 1
                    if yr < simplex_vals[0]:
                        xe = centroid + gamma * (xr - centroid)
                        xe = np.clip(xe, lb, ub)
                        ye = func(xe)
                        remaining -= 1
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
                        remaining -= 1
                        if yc < simplex_vals[-1]:
                            simplex[-1] = xc
                            simplex_vals[-1] = yc
                        else:
                            for i in range(1, dim + 1):
                                if remaining <= 0:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                y = func(simplex[i])
                                remaining -= 1
                                simplex_vals[i] = y
                                if y < best_val:
                                    best_val = y
                                    best_x = simplex[i].copy()
                                    report_best(best_val, best_x)

                    idx_min = np.argmin(simplex_vals)
                    if simplex_vals[idx_min] < best_val:
                        best_val = simplex_vals[idx_min]
                        best_x = simplex[idx_min].copy()
                        report_best(best_val, best_x)

                    if remaining <= 0:
                        break

                if remaining > 0:
                    x = best_x + 0.01 * (ub - lb) * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    remaining -= 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                else:
                    break

        return best_val, best_x