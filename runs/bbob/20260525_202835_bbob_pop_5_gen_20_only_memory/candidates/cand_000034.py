import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        best_val = np.inf
        best_x = None
        total_evals = 0

        if budget < 50:
            num_restarts = 1
        else:
            num_restarts = 3
        sub_budget = budget // num_restarts
        remainder = budget % num_restarts

        for restart in range(num_restarts):
            if restart < remainder:
                r_budget = sub_budget + 1
            else:
                r_budget = sub_budget
            if r_budget <= 0:
                continue
            sub_seed = self.rng.integers(0, 2**31)
            local_rng = np.random.default_rng(sub_seed)
            r_best_val, r_best_x, r_evals = self._run_restart(func, lb, ub, dim, r_budget, local_rng)
            total_evals += r_evals
            if r_best_val < best_val:
                best_val = r_best_val
                best_x = r_best_x.copy()
                report_best(best_val, best_x)
        return best_val, best_x

    def _run_restart(self, func, lb, ub, dim, budget, rng):
        n = max(4, min(5*dim, budget//2))
        if n <= 1:
            x = rng.uniform(lb, ub)
            val = func(x)
            return val, x, 1
        F = 0.8
        CR = 0.9
        evals = 0

        pop = np.empty((n, dim))
        for d in range(dim):
            cuts = np.linspace(lb[d], ub[d], n+1)
            u = rng.uniform(0, 1, n)
            perm = rng.permutation(n)
            pop[perm, d] = cuts[:-1] + u * (cuts[1:] - cuts[:-1])
        pop = np.clip(pop, lb, ub)

        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf

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

        while evals < budget:
            for i in range(n):
                if evals >= budget:
                    break
                idx = list(range(n))
                idx.remove(i)
                a, b, c = rng.choice(idx, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross = rng.random(dim) < CR
                if not np.any(cross):
                    cross[rng.integers(dim)] = True
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

        # Local search on best solution
        if evals < budget and best_x is not None:
            step = 0.1 * (ub - lb)
            max_iters = 3
            for _ in range(max_iters):
                if evals >= budget:
                    break
                improved_any = False
                order = rng.permutation(dim)
                for i in order:
                    if evals >= budget:
                        break
                    # positive direction
                    x_new = best_x.copy()
                    x_new[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                        step[i] *= 2.0
                        improved_any = True
                        continue
                    # negative direction
                    if evals >= budget:
                        break
                    x_new = best_x.copy()
                    x_new[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                        step[i] *= 2.0
                        improved_any = True
                    else:
                        step[i] *= 0.5
                if not improved_any:
                    if evals >= budget:
                        break
                    d = rng.normal(0, 1, dim)
                    d = d / np.linalg.norm(d)
                    scale = np.mean(step) * 0.5
                    x_new = np.clip(best_x + scale * d, lb, ub)
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
        return best_val, best_x, evals