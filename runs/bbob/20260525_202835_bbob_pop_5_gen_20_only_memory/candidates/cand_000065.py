import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(10, min(4 * dim, budget // 3))
        self.stall_limit = max(10, budget // 20)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        rng = self.rng
        budget = self.budget
        stall_limit = self.stall_limit

        pop = rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        for i in range(popsize):
            if evals >= budget:
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

        generations_since_improvement = 0
        eval_at_last_local = 0
        step = 0.1 * (ub - lb)

        while evals < budget:
            F = rng.uniform(0.5, 1.0)
            CR = rng.uniform(0.5, 1.0)
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                r1, r2 = rng.choice(candidates, 2, replace=False)
                idx_best = np.argmin(fitness)
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
                        generations_since_improvement = 0
                    else:
                        generations_since_improvement += 1
                else:
                    generations_since_improvement += 1

            if evals - eval_at_last_local >= 2 * dim and evals < budget:
                x_best = best_x.copy()
                f_best = best_val
                for i in range(dim):
                    if evals >= budget:
                        break
                    x_new = x_best.copy()
                    x_new[i] = np.clip(x_best[i] + step[i], lb[i], ub[i])
                    val = func(x_new)
                    evals += 1
                    if val < f_best:
                        f_best = val
                        x_best[i] = x_new[i]
                        if val < best_val:
                            best_val = val
                            best_x = x_best.copy()
                            report_best(best_val, best_x)
                        continue
                    if evals >= budget:
                        break
                    x_new = x_best.copy()
                    x_new[i] = np.clip(x_best[i] - step[i], lb[i], ub[i])
                    val = func(x_new)
                    evals += 1
                    if val < f_best:
                        f_best = val
                        x_best[i] = x_new[i]
                        if val < best_val:
                            best_val = val
                            best_x = x_best.copy()
                            report_best(best_val, best_x)
                if f_best < best_val:
                    best_val = f_best
                    best_x = x_best.copy()
                    report_best(best_val, best_x)
                eval_at_last_local = evals

            if generations_since_improvement > stall_limit and evals < budget:
                n_restart = popsize // 2
                restart_indices = rng.choice(popsize, n_restart, replace=False)
                for idx in restart_indices:
                    if evals >= budget:
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
                generations_since_improvement = 0

        return best_val, best_x