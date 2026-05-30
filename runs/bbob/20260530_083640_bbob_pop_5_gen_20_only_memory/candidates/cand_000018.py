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

        pop_size = max(2 * dim, min(10 + int(dim**0.5), budget // 2))
        pop_size = min(pop_size, budget)

        best_val = np.inf
        best_x = None
        evals = 0

        max_no_improve = max(1, int(0.1 * budget))

        while evals < budget:
            # Latin Hypercube initial population
            lhs = self._latin_hypercube(pop_size, dim, rng)
            bounds = np.array([lb, ub]).T
            pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

            pop_fitness = np.full(pop_size, np.inf)
            for i in range(pop_size):
                if evals >= budget:
                    break
                x = np.clip(pop[i], lb, ub)
                val = func(x)
                evals += 1
                pop_fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

            # DE parameters
            F = 0.5
            CR = 0.9
            no_improve_count = 0
            while evals < budget and no_improve_count < max_no_improve:
                prev_best = best_val
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    # Mutation
                    indices = [j for j in range(pop_size) if j != i]
                    rng.shuffle(indices)
                    a, b, c = indices[:3]
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    # Crossover
                    j_rand = rng.randint(dim)
                    trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                    trial = np.clip(trial, lb, ub)
                    # Evaluation
                    val = func(trial)
                    evals += 1
                    if val < pop_fitness[i]:
                        pop[i] = trial
                        pop_fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                if best_val < prev_best:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            # After DE, if budget remains, do coordinate descent
            if evals >= budget:
                break
            remaining = budget - evals
            if remaining <= 0:
                break
            # Coordinate descent parameters
            step = 0.1 * (ub - lb)
            max_outer_iters = min(remaining, dim * 5)
            for _ in range(max_outer_iters):
                if evals >= budget:
                    break
                improved = False
                for i in range(dim):
                    if evals >= budget:
                        break
                    # Try positive step
                    x_new = best_x.copy()
                    x_new[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                    val = func(x_new)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                        improved = True
                        continue
                    # Try negative step
                    x_new[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                    val = func(x_new)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                        improved = True
                if not improved:
                    step *= 0.9  # reduce step size
                if evals >= budget:
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