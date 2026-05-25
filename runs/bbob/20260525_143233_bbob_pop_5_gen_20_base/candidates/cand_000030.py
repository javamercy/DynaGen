import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(5 * dim, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget
        self.restart_threshold = max(5, dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        evals = 0
        best_val = np.inf
        best_x = None

        pop = np.random.uniform(lb, ub, (pop_size, dim)) if pop_size > 0 else np.empty((0, dim))
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
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

        if evals == 0:
            return best_val, best_x

        CR = 0.9
        no_improve = 0
        while evals < self.budget:
            improved = False
            F = np.random.uniform(0.5, 1.0)
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c, d, e = np.random.choice(idxs, 5, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
                mutant = np.clip(mutant, lb, ub)
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            if improved:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.restart_threshold:
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= self.budget:
                        break
                    x = new_pop[i]
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                no_improve = 0
        return best_val, best_x