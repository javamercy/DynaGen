import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.local_budget = max(10, int(0.2 * budget))
        self.de_budget = budget - self.local_budget
        self.popsize = max(4, min(4 * dim, self.de_budget // 2))
        self.F_init = 0.8
        self.F_final = 0.2
        self.CR_init = 0.2
        self.CR_final = 0.9

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # evaluate initial population
        for i in range(popsize):
            if evaluations >= self.de_budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # DE loop with adaptive parameters
        max_gens = self.de_budget // popsize
        gen = 0
        while evaluations < self.de_budget:
            if max_gens > 0:
                progress = min(1.0, gen / max_gens)
            else:
                progress = 1.0
            F = self.F_init + (self.F_final - self.F_init) * progress
            CR = self.CR_init + (self.CR_final - self.CR_init) * progress
            for i in range(popsize):
                if evaluations >= self.de_budget:
                    break
                indices = list(range(popsize))
                indices.remove(i)
                a, b, c = self.rng.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
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
            gen += 1
        # Local search: random perturbation around best
        step_size = (ub - lb) * 0.01
        while evaluations < self.budget:
            trial = best_x + self.rng.normal(0, step_size)
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evaluations += 1
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                report_best(best_val, best_x)
        return best_val, best_x