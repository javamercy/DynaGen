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

        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # Phase 1: Simulated Annealing
        budget_sa = max(1, int(0.8 * budget))
        max_iter_sa = budget_sa - 1
        if max_iter_sa > 0:
            T0 = 1.0
            T_end = 1e-4
            step0 = 0.1 * (ub - lb)
            step_end = 1e-6 * (ub - lb)
            current_x = best_x.copy()
            current_val = best_val
            for i in range(max_iter_sa):
                if evals >= budget_sa:
                    break
                t = i / max_iter_sa
                T = T0 * (T_end / T0) ** t
                step = step0 * (step_end / step0) ** t
                candidate = current_x + step * rng.randn(dim)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                delta = val - current_val
                if delta < 0:
                    current_x = candidate
                    current_val = val
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                else:
                    if rng.rand() < np.exp(-delta / T):
                        current_x = candidate
                        current_val = val
                if evals >= budget:
                    return best_val, best_x

        # Phase 2: Nelder-Mead local search
        remaining = budget - evals
        if remaining <= 0:
            return best_val, best_x

        # Build simplex around best point
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

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = 100
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