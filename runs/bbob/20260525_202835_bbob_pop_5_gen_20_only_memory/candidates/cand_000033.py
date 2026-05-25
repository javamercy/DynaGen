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

        # Latin Hypercube Sampling
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
                if len(idx) < 3:
                    break
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

        # local refinement via random perturbations
        step = 0.1 * (ub - lb)
        while evals < budget:
            d = self.rng.normal(0, 1, dim)
            d = d / (np.linalg.norm(d) + 1e-12)
            x_new = best_x + step * d
            x_new = np.clip(x_new, lb, ub)
            val = func(x_new)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x_new.copy()
                report_best(best_val, best_x)

        return best_val, best_x