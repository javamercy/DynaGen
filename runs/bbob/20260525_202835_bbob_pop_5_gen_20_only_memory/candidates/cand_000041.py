import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(5 * dim, budget // 3))
        self.F = 0.5
        self.CR = 0.5

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        budget = self.budget
        evals = 0

        # Latin Hypercube Sampling for initial population
        pop = np.empty((n, dim))
        for d in range(dim):
            cuts = np.linspace(lb[d], ub[d], n+1)
            u = self.rng.uniform(0, 1, n)
            perm = self.rng.permutation(n)
            pop[perm, d] = cuts[:-1] + u * (cuts[1:] - cuts[:-1])
        pop = np.clip(pop, lb, ub)

        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf

        # initial evaluation
        for i in range(n):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # DE loop
        while evals < budget:
            for i in range(n):
                if evals >= budget:
                    break
                idx = list(range(n))
                idx.remove(i)
                a, b, c = self.rng.choice(idx, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross = self.rng.random(dim) < self.CR
                if not np.any(cross):
                    cross[self.rng.integers(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Nelder-Mead local search
        if evals < budget:
            best_x, best_val, used = self._nelder_mead(func, best_x, best_val, lb, ub, budget - evals)
            evals += used
        return best_val, best_x

    def _nelder_mead(self, func, x0, f0, lb, ub, budget_left):
        dim = self.dim
        rng = self.rng
        delta = 0.05 * (ub - lb)
        delta = np.maximum(delta, 1e-8 * np.ones(dim))
        x = np.zeros((dim+1, dim))
        x[0] = x0.copy()
        for i in range(dim):
            point = x0.copy()
            point[i] = np.clip(x0[i] + delta[i], lb[i], ub[i])
            x[i+1] = point
        f = np.full(dim+1, np.inf)
        f[0] = f0
        evals = 0
        for i in range(1, dim+1):
            if evals >= budget_left:
                break
            f[i] = func(x[i])
            evals += 1
        idx = np.argsort(f)
        x = x[idx]
        f = f[idx]
        best_val = f[0]
        best_x = x[0].copy()
        if best_val < f0:
            report_best(best_val, best_x)
        rho = 1.0
        chi = 2.0
        psi = 0.5
        sigma = 0.5
        while evals < budget_left:
            centroid = np.mean(x[:-1], axis=0)
            xr = centroid + rho * (centroid - x[-1])
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evals += 1
            if fr < f[0]:
                xe = centroid + chi * rho * (centroid - x[-1])
                xe = np.clip(xe, lb, ub)
                fe = func(xe)
                evals += 1
                if fe < fr:
                    x[-1] = xe
                    f[-1] = fe
                else:
                    x[-1] = xr
                    f[-1] = fr
            elif fr < f[-2]:
                x[-1] = xr
                f[-1] = fr
            else:
                if fr < f[-1]:
                    xc = centroid + psi * (xr - centroid)
                else:
                    xc = centroid - psi * (centroid - x[-1])
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evals += 1
                if fc < f[-1]:
                    x[-1] = xc
                    f[-1] = fc
                else:
                    for i in range(1, dim+1):
                        if evals >= budget_left:
                            break
                        x[i] = x[0] + sigma * (x[i] - x[0])
                        x[i] = np.clip(x[i], lb, ub)
                        f[i] = func(x[i])
                        evals += 1
                    x[0] = x[0].copy()
            idx = np.argsort(f)
            x = x[idx]
            f = f[idx]
            if f[0] < best_val:
                best_val = f[0]
                best_x = x[0].copy()
                report_best(best_val, best_x)
        return best_x, best_val, evals