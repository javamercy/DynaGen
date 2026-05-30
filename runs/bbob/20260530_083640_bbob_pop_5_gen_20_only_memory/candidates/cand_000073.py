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

        # Population size for DE
        pop_size = min(budget // 2, max(4 * dim, min(15 + int(dim**0.5), budget // 2)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget)

        # Latin Hypercube initial population
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial points
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

        # Reserve for local search
        reserve = max(2 * dim, 30)
        reserve = min(reserve, budget - evals)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        # Adaptive DE parameters (mutated schedule)
        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.8 - 0.3 * frac  # 0.8 -> 0.5
            CR = 0.7 + 0.3 * frac  # 0.7 -> 1.0 but capped to 1.0
            CR = min(CR, 1.0)
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

        # Local search: coordinate descent on best point (mutated parameters)
        remaining = budget - evals
        if remaining > 0:
            step0 = 0.2 * (ub - lb)  # larger initial step
            x = best_x.copy()
            y = best_val
            max_iter = min(remaining // (2 * dim), 50)  # fewer iterations
            for it in range(max_iter):
                if evals >= budget:
                    break
                step = step0 * (0.8 ** it)  # faster decay
                dims = list(range(dim))
                rng.shuffle(dims)
                for d in dims:
                    if evals >= budget:
                        break
                    # positive step
                    x_candidate = x.copy()
                    x_candidate[d] = min(ub[d], max(lb[d], x_candidate[d] + step[d]))
                    val = func(x_candidate)
                    evals += 1
                    if val < y:
                        x = x_candidate
                        y = val
                        if y < best_val:
                            best_val = y
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        continue
                    # negative step
                    x_candidate = x.copy()
                    x_candidate[d] = min(ub[d], max(lb[d], x_candidate[d] - step[d]))
                    val = func(x_candidate)
                    evals += 1
                    if val < y:
                        x = x_candidate
                        y = val
                        if y < best_val:
                            best_val = y
                            best_x = x.copy()
                            report_best(best_val, best_x)
        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs