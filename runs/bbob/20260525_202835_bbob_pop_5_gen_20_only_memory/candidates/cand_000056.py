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
        rng = self.rng

        # DE parameters
        popsize = max(4, min(4 * dim, budget // 4))
        F = 0.8
        CR = 0.9

        # Phase 1: DE
        pop = rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0

        # Initial evaluation of population
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

        if best_x is None:
            x = rng.uniform(lb, ub)
            val = func(x)
            evaluations += 1
            best_val = val
            best_x = x.copy()
            report_best(best_val, best_x)

        # DE iterations up to 2/3 of budget
        de_budget = int(budget * 2 / 3)
        while evaluations < de_budget and evaluations < budget:
            for i in range(popsize):
                if evaluations >= de_budget or evaluations >= budget:
                    break
                idx_best = np.argmin(fitness)
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

        # Phase 2: Local random search around best with shrinking step
        scale = np.max(ub - lb)
        while evaluations < budget:
            step = scale * (1 - evaluations / budget) * 0.1
            trial = best_x + rng.normal(0, step, size=dim)
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evaluations += 1
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                report_best(best_val, best_x)

        return best_val, best_x