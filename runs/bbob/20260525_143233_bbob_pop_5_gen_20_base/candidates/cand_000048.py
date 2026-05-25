import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(5, min(10 * dim, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget
        self.restart_threshold = 20

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        evals = 0
        best_val = np.inf
        best_x = None

        if pop_size <= 0:
            x = np.random.uniform(lb, ub, dim)
            best_val = func(x)
            best_x = x.copy()
            report_best(best_val, best_x)
            evals = 1
            while evals < self.budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = np.random.uniform(lb, ub, (pop_size, dim))
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

        F = 0.7
        CR = 0.9
        no_improve = 0
        gen = 0
        while evals < self.budget:
            if pop_size == 0:
                break
            improved = False
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
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
                # Keep best individual only
                best_idx = np.argmin(fitness)
                new_pop = np.empty_like(pop)
                new_fitness = np.full(pop_size, np.inf)
                new_pop[0] = pop[best_idx].copy()
                new_fitness[0] = fitness[best_idx]
                for i in range(1, pop_size):
                    if evals >= self.budget:
                        break
                    x = np.random.uniform(lb, ub, dim)
                    new_pop[i] = x
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
            gen += 1
            if evals >= self.budget:
                break
        return best_val, best_x