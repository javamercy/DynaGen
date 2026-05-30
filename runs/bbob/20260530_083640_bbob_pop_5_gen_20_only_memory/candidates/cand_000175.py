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

        # Population size: small for more local search budget
        pop_size = max(2 * dim, 12)
        if pop_size >= budget // 2:
            pop_size = budget // 4
        if pop_size < 2:
            pop_size = 2

        # LHS initialization
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs
        pop = np.clip(pop, lb, ub)

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = pop[i]
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Reserve budget for local search (70% of remaining after initial evaluations)
        local_budget = int(0.7 * (budget - evals))
        de_budget = (budget - evals) - local_budget
        max_gen = de_budget // pop_size
        max_gen = max(0, min(max_gen, 100))

        # DE with current-to-best mutation
        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.8 - 0.4 * frac
            CR = 0.9 - 0.4 * frac
            for i in range(pop_size):
                # current-to-best: base = pop[i] + F*(best - pop[i]) + F*(pop[r1] - pop[r2])
                r1, r2 = rng.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
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

        # Coordinate descent with adaptive step sizes
        remaining = budget - evals
        if remaining > 0:
            x = best_x.copy()
            y = best_val
            step_size = 0.1 * (ub - lb)
            min_step = 1e-5 * (ub - lb)
            for iteration in range(100):
                if evals >= budget:
                    break
                improved = False
                for d in range(dim):
                    if evals >= budget:
                        break
                    # try positive step
                    x_candidate = x.copy()
                    x_candidate[d] = np.clip(x[d] + step_size[d], lb[d], ub[d])
                    val = func(x_candidate)
                    evals += 1
                    if val < y:
                        x = x_candidate
                        y = val
                        if y < best_val:
                            best_val = y
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        improved = True
                        continue
                    # try negative step
                    x_candidate = x.copy()
                    x_candidate[d] = np.clip(x[d] - step_size[d], lb[d], ub[d])
                    val = func(x_candidate)
                    evals += 1
                    if val < y:
                        x = x_candidate
                        y = val
                        if y < best_val:
                            best_val = y
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        improved = True
                if not improved:
                    step_size = step_size / 2.0
                    if np.all(step_size < min_step):
                        break
        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs