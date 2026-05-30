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

        pop_size = min(budget // 2, max(3 * dim, min(15 + int(dim**0.5), budget // 2)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget)

        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        reserve = max(2 * dim, 20)
        reserve = min(reserve, budget - evals)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                if evals >= budget:
                    return best_val, best_x

        remaining = budget - evals
        if remaining > 0:
            x_curr = best_x.copy()
            f_curr = best_val
            step_size = 0.1 * (ub - lb)
            min_step = 1e-5 * (ub - lb)
            max_iter = remaining
            iter_count = 0
            while evals < budget and iter_count < max_iter and np.any(step_size > min_step):
                improved = False
                for i in range(dim):
                    if evals >= budget:
                        break
                    x_new = x_curr.copy()
                    x_new[i] = min(ub[i], max(lb[i], x_new[i] + step_size[i]))
                    val_new = func(x_new)
                    evals += 1
                    if val_new < f_curr:
                        x_curr = x_new
                        f_curr = val_new
                        improved = True
                        if f_curr < best_val:
                            best_val = f_curr
                            best_x = x_curr.copy()
                            report_best(best_val, best_x)
                        continue
                    x_new = x_curr.copy()
                    x_new[i] = min(ub[i], max(lb[i], x_new[i] - step_size[i]))
                    val_new = func(x_new)
                    evals += 1
                    if val_new < f_curr:
                        x_curr = x_new
                        f_curr = val_new
                        improved = True
                        if f_curr < best_val:
                            best_val = f_curr
                            best_x = x_curr.copy()
                            report_best(best_val, best_x)
                if not improved:
                    step_size *= 0.5
                iter_count += 1
        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs