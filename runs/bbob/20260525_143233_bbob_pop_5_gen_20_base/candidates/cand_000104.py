import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(2 * dim, budget // 10))
        self.restart_threshold = max(5, dim)

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=np.float64)
        ub = np.array(func.bounds.ub, dtype=np.float64)
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        # initial best
        best_x = rng.uniform(lb, ub, dim).astype(np.float64)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        if budget <= 10:
            for _ in range(budget - 1):
                x = rng.uniform(lb, ub, dim).astype(np.float64)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # population initialization
        pop = rng.uniform(lb, ub, (pop_size, dim)).astype(np.float64)
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        CR = 0.3
        no_improve = 0
        local_search_threshold = max(5, dim // 2)

        while evals < budget:
            improved_in_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                F = rng.uniform(0.5, 0.8)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved_in_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if not improved_in_gen:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= local_search_threshold and evals < budget:
                local_budget = min(budget - evals, max(10, 2 * dim))
                scale = 0.1 * (ub - lb)
                for _ in range(local_budget):
                    if evals >= budget:
                        break
                    x = best_x + scale * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                no_improve = 0

        return best_val, best_x