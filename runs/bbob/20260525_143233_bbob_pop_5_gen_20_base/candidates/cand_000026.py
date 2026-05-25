import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        np.random.seed(seed)
        self.pop_size = max(4, min(4*dim, budget // 2))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        # Initial best
        best_x = np.random.uniform(lb, ub, dim)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)
        if evals >= self.budget:
            return best_val, best_x
        # Initialize population, include best
        pop = np.random.uniform(lb, ub, (pop_size-1, dim))
        pop = np.vstack([best_x, pop])
        fitness = np.full(pop_size, np.inf)
        fitness[0] = best_val
        for i in range(1, pop_size):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        F = 0.5
        CR = 0.9
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Select three distinct random indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
        return best_val, best_x