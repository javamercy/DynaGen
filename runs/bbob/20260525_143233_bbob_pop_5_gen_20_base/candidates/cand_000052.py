import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.pop_size = max(self.pop_size, 1)
        self.restart_threshold = max(5, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        rng = self.rng

        if pop_size <= 1:
            best_x = rng.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
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
        no_improve = 0

        while evals < budget:
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                mutant = best_x + F * (pop[r1] - pop[r2])
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
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # local search around best
            if evals < budget:
                local_evals = min(2, budget - evals)
                for _ in range(local_evals):
                    sigma = 0.01 * (ub - lb)
                    x = best_x + sigma * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            if improved_this_gen:
                F *= 1.1
                F = min(F, 0.9)
                no_improve = 0
            else:
                F *= 0.9
                F = max(F, 0.1)
                no_improve += 1

            if no_improve >= self.restart_threshold:
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_focused = max(1, int(0.3 * pop_size))
                    for j in range(num_focused):
                        new_pop[j] = best_x + 0.1 * rng.randn(dim) * (ub - lb)
                        new_pop[j] = np.clip(new_pop[j], lb, ub)
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
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
                F = 0.5
                no_improve = 0

        return best_val, best_x