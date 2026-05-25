import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.pop_size = max(self.pop_size, 1)
        self.max_generations = (budget - self.pop_size) // self.pop_size if self.pop_size > 0 else 0
        self.restart_threshold = max(5, 2 * dim)

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
        generation = 0
        while evals < self.budget and generation < self.max_generations:
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 5:
                    # fallback to rand/1 if not enough candidates
                    if len(candidates) < 3:
                        continue
                    a, b, c = np.random.choice(candidates, size=3, replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    # rand/2 mutation
                    idx = np.random.choice(candidates, size=5, replace=False)
                    a, b, c, d, e = idx
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
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
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
                F = 0.5
                no_improve = 0
            generation += 1
        return best_val, best_x