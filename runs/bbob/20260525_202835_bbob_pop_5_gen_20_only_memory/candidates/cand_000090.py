import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(10, min(2 * dim, self.budget // 2))
        self.stall_limit = max(10, budget // 10)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        for i in range(popsize):
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        if best_x is None:
            best_x = self.rng.uniform(lb, ub)
            best_val = func(best_x)
            evals += 1
            report_best(best_val, best_x)
        stall = 0
        while evals < self.budget:
            for i in range(popsize):
                if evals >= self.budget:
                    break
                idx_best = np.argmin(fitness)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + 0.8 * (pop[idx_best] - pop[i]) + 0.8 * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.random(dim) < 0.9
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stall = 0
                    else:
                        stall += 1
                else:
                    stall += 1
            if stall > self.stall_limit:
                n_restart = popsize // 2
                restart_indices = self.rng.choice(popsize, n_restart, replace=False)
                for idx in restart_indices:
                    if evals >= self.budget:
                        break
                    new_x = self.rng.uniform(lb, ub)
                    val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stall = 0
        return best_val, best_x