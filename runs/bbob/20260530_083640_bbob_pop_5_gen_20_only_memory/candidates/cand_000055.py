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

        # Phase 1: DE with scheduled parameters
        pop_size = max(4, min(10, budget // 10))
        de_budget = max(2 * (pop_size + 1), budget // 3)
        de_evals = 0

        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
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
            # Schedule CR and F linearly decreasing
            alpha = gen / max_gen if max_gen > 0 else 0
            CR = 0.9 - 0.7 * alpha  # from 0.9 to 0.2
            F = 0.9 - 0.7 * alpha   # same schedule
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

        # Phase 2: Nelder-Mead with adaptive step size
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
            step_magnitude = 0.01  # initial
            while remaining > 0:
                # Build simplex
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = best_x.copy()
                for i in range(dim):
                    step = step_magnitude * (ub[i] - lb[i])
                    if step == 0:
                        step = step_magnitude * 0.001
                    x = best_x.copy()
                    x[i] = np.clip(x[i] + step, lb[i], ub[i])
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
                improved_in_nm = False
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
                        improved_in_nm = True
                    elif yr < simplex_vals[-2]:
                        simplex[-1] = xr
                        simplex_vals[-1] = yr
                        improved_in_nm = True
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
                            improved_in_nm = True
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

                if not improved_in_nm:
                    step_magnitude *= 0.5
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