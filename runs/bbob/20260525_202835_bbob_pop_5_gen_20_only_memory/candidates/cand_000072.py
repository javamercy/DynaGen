import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = min(max(20, 4 * dim), budget // 2)
        if self.popsize < 4:
            self.popsize = min(4, budget)
        self.stall_limit = max(10, budget // 10)

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
        # initial evaluation
        for i in range(popsize):
            if evaluations >= self.budget:
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
            # fallback: evaluate a random point
            x = self.rng.uniform(lb, ub)
            if evaluations < self.budget:
                val = func(x)
                evaluations += 1
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # main loop
        stall_count = 0
        while evaluations < self.budget:
            F = 0.8
            CR = 0.9
            improved = False
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                idx_best = np.argmin(fitness)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
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
                    improved = True
            if improved:
                stall_count = 0
            else:
                stall_count += 1
            if stall_count >= self.stall_limit:
                # restart: keep best individual
                best_idx = np.argmin(fitness)
                best_individual = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                pop = self.rng.uniform(lb, ub, size=(popsize, dim))
                pop[0] = best_individual
                fitness = np.full(popsize, np.inf)
                fitness[0] = best_fit
                for i in range(1, popsize):
                    if evaluations >= self.budget:
                        break
                    x = pop[i]
                    val = func(x)
                    evaluations += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                stall_count = 0
        return best_val, best_x