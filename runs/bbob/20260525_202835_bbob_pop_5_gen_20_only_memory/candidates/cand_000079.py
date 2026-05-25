import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(10, min(4 * dim, budget // 4))
        self.stall_limit = max(5, budget // 25)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        rng = self.rng
        pop = rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        for i in range(popsize):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        if best_x is None:
            best_x = rng.uniform(lb, ub)
            best_val = func(best_x)
            evals += 1
            report_best(best_val, best_x)
        gen = 0
        stall = 0
        while evals < self.budget:
            gen += 1
            F = rng.uniform(0.5, 1.0)
            CR = rng.uniform(0.5, 1.0)
            for i in range(popsize):
                if evals >= self.budget:
                    break
                idx_best = np.argmin(fitness)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.integers(dim)] = True
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
            if gen % 5 == 0 and evals < self.budget:
                step_size = 0.1 * (ub - lb) * (1 - evals / self.budget)
                new_x = best_x + rng.normal(0, step_size)
                new_x = np.clip(new_x, lb, ub)
                val = func(new_x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                    stall = 0
            if evals < self.budget:
                idx = rng.integers(popsize)
                new_x = rng.uniform(lb, ub)
                val = func(new_x)
                evals += 1
                pop[idx] = new_x
                fitness[idx] = val
                if val < best_val:
                    best_val = val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                    stall = 0
            if stall > self.stall_limit:
                n_restart = popsize // 2
                best_idx = np.argmin(fitness)
                candidates_indices = [j for j in range(popsize) if j != best_idx]
                if len(candidates_indices) < n_restart:
                    restart_indices = rng.choice(candidates_indices, len(candidates_indices), replace=False)
                else:
                    restart_indices = rng.choice(candidates_indices, n_restart, replace=False)
                for idx in restart_indices:
                    if evals >= self.budget:
                        break
                    new_x = rng.uniform(lb, ub)
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