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

        # Phase 1: Cross-Entropy Method
        sample_size = min(10 * dim, budget // 3)
        sample_size = max(sample_size, 10)
        elite_frac = 0.25
        mean = (lb + ub) / 2.0
        diff = ub - lb
        std = 0.4 * diff
        min_std = 1e-6 * diff

        best_x = mean.copy()
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        cem_budget = int(0.7 * budget)
        while evals < cem_budget and evals < budget:
            remaining = min(cem_budget - evals, budget - evals)
            if remaining <= 0:
                break
            n_samples = min(sample_size, remaining)
            samples = rng.randn(n_samples, dim) * std + mean
            samples = np.clip(samples, lb, ub)
            vals = np.full(n_samples, np.inf)
            for i in range(n_samples):
                if evals >= budget or evals >= cem_budget:
                    break
                x = samples[i]
                val = func(x)
                evals += 1
                vals[i] = val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            if evals >= budget or evals >= cem_budget:
                break
            idx = np.argsort(vals)
            n_elite = max(1, int(elite_frac * n_samples))
            elite = samples[idx[:n_elite]]
            new_mean = np.mean(elite, axis=0)
            new_std = np.std(elite, axis=0) + 1e-10
            alpha = 0.7
            mean = (1 - alpha) * mean + alpha * new_mean
            std = (1 - alpha) * std + alpha * new_std
            std = np.maximum(std, min_std)
            if np.mean(std) < 1e-3 * np.mean(diff):
                mean = lb + rng.rand(dim) * diff
                std = 0.4 * diff

        # Phase 2: Nelder-Mead Local Search
        if evals < budget:
            frac_nm = 1.0 - evals / budget
            step = (0.2 - 0.15 * frac_nm) * diff
            step = np.clip(step, 0.01 * diff, None)
            simplex = np.tile(best_x, (dim + 1, 1))
            for i in range(dim):
                simplex[i + 1, i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
            fvals = np.full(dim + 1, np.inf)
            fvals[0] = best_val
            for i in range(1, dim + 1):
                if evals >= budget:
                    break
                x = np.clip(simplex[i], lb, ub)
                val = func(x)
                evals += 1
                fvals[i] = val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
            while evals < budget:
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + rho * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget:
                    break
                fr = func(xr)
                evals += 1
                if fr < fvals[0]:
                    xe = centroid + chi * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= budget:
                        break
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = centroid + psi * (xr - centroid)
                    else:
                        xc = centroid - psi * (centroid - simplex[-1])
                    xc = np.clip(xc, lb, ub)
                    if evals >= budget:
                        break
                    fc = func(xc)
                    evals += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            if evals >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            val_i = func(simplex[i])
                            evals += 1
                            fvals[i] = val_i
                            if val_i < best_val:
                                best_val = val_i
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)
                        if evals >= budget:
                            break
                current_best = np.argmin(fvals)
                if fvals[current_best] < best_val:
                    best_val = fvals[current_best]
                    best_x = simplex[current_best].copy()
                    report_best(best_val, best_x)

        return best_val, best_x