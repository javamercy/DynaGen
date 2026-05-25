import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.restart_threshold = max(5, 2 * dim)
        self.local_search_evals_per_gen = 2

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

        if pop_size <= 0:
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
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
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        F = 0.5
        CR = 0.9
        F_min = 0.1
        F_max = 0.9
        CR_min = 0.2
        CR_max = 0.9
        F_adapt_up = 1.1
        F_adapt_down = 0.9
        CR_adapt_up = 0.05
        CR_adapt_down = 0.05

        no_improve = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0
        gen = 0

        while evals < budget and gen < max_gen:
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= budget:
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
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if improved_this_gen:
                F = min(F * F_adapt_up, F_max)
                CR = min(CR + CR_adapt_up, CR_max)
                no_improve = 0
            else:
                F = max(F * F_adapt_down, F_min)
                CR = max(CR - CR_adapt_down, CR_min)
                no_improve += 1

            if evals < budget:
                local_evals = min(self.local_search_evals_per_gen, budget - evals)
                for _ in range(local_evals):
                    sigma = 0.01 * (ub - lb)
                    x = best_x + sigma * np.random.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            if no_improve >= self.restart_threshold:
                new_pop = np.empty_like(pop)
                new_fitness = np.full(pop_size, np.inf)
                new_pop[0] = best_x.copy()
                new_fitness[0] = best_val
                focused_count = max(0, int(0.3 * pop_size) - 1)
                for j in range(1, 1 + focused_count):
                    sigma = 0.1 * (ub - lb)
                    x = best_x + sigma * np.random.randn(dim)
                    x = np.clip(x, lb, ub)
                    new_pop[j] = x
                    if evals < budget:
                        val = func(x)
                        evals += 1
                        new_fitness[j] = val
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                for j in range(1 + focused_count, pop_size):
                    if evals >= budget:
                        break
                    x = np.random.uniform(lb, ub, dim)
                    new_pop[j] = x
                    val = func(x)
                    evals += 1
                    new_fitness[j] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                F = 0.5
                CR = 0.9
                no_improve = 0
            gen += 1

        return best_val, best_x