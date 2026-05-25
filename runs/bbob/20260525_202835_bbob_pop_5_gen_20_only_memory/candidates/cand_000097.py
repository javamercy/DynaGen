import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(4 * dim, budget // 2))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        budget = self.budget
        rng = self.rng

        # Latin hypercube initialization
        samples = np.zeros((popsize, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, popsize + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            samples[:, d] = points
        pop = lb + samples * (ub - lb)

        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0

        for i in range(popsize):
            if evaluations >= budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        max_gen_evals = budget - evaluations
        generations = max(1, max_gen_evals // popsize)
        evals_per_gen = popsize

        for gen in range(generations):
            if evaluations >= budget:
                break
            # Schedule F and CR linearly over generations
            frac = gen / generations
            F = 0.9 - frac * (0.9 - 0.3)  # from 0.9 to 0.3
            CR = 0.1 + frac * (0.9 - 0.1)  # from 0.1 to 0.9

            idx_best = np.argmin(fitness)
            for i in range(popsize):
                if evaluations >= budget:
                    break
                r1, r2 = rng.choice([j for j in range(popsize) if j != i], 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x