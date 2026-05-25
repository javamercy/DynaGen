import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.restart_threshold = max(5, dim // 2)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        if pop_size <= 0:
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
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
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        generation = 0
        max_gen = (self.budget - evals) // pop_size
        while evals < self.budget and generation < max_gen:
            # parameter schedule based on generation fraction
            frac = generation / max_gen if max_gen > 0 else 1.0
            F = 1.0 - 0.8 * frac  # 1.0 -> 0.2
            CR = 0.2 + 0.7 * frac  # 0.2 -> 0.9
            local_sigma = 0.05 - 0.049 * frac  # 0.05 -> 0.001
            # mutation using DE/best/1
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                crossover = np.random.rand(dim) < CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, dim)] = True
                trial = np.where(crossover, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # local search around best
            if evals < self.budget:
                local_evals = min(5, self.budget - evals)
                for _ in range(local_evals):
                    x = best_x + local_sigma * np.random.randn(dim) * (ub - lb)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
            # check stagnation
            no_improve = 0
            if evals > 0 and np.min(fitness) == best_val:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.restart_threshold:
                # focused restart: 50% around best, 50% random
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_focused = int(0.5 * pop_size)
                    for j in range(num_focused):
                        new_pop[j] = best_x + 0.1 * np.random.randn(dim) * (ub - lb)
                        new_pop[j] = np.clip(new_pop[j], lb, ub)
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= self.budget:
                        break
                    x = new_pop[i].copy()
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
            generation += 1
        return best_val, best_x