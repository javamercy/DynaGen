import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        pop_size = max(4, min(20, budget // 10))
        local_budget = max(1, budget // 5)
        global_budget = budget - local_budget
        n_generations = max(1, global_budget // pop_size)

        best_val = float('inf')
        best_x = None
        evals = 0

        # initialization
        population = rng.uniform(lb, ub, size=(pop_size, dim))
        for i in range(pop_size):
            if evals >= budget:
                break
            x = population[i]
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        F = 0.8
        CR = 0.9
        for gen in range(n_generations):
            if evals >= budget:
                break
            # DE mutation and crossover
            new_pop = np.empty_like(population)
            for i in range(pop_size):
                if evals >= budget:
                    break
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = rng.choice(indices, size=3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                new_pop[i] = trial
            population = new_pop

        # local refinement
        scale0 = np.linalg.norm(ub - lb) * 0.05
        for _ in range(local_budget):
            if evals >= budget:
                break
            step = rng.normal(0, scale0, size=dim)
            candidate = best_x + step
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)

        return best_val, best_x