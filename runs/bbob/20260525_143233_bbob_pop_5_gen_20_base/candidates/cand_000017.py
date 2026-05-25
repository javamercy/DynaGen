import numpy as np
class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # Population size: at least 4*dim or 20, but not more than budget
        pop_size = max(4 * dim, 20)
        pop_size = min(pop_size, budget)
        if pop_size < 1:
            pop_size = 1
        self.pop_size = pop_size
        if pop_size > 0:
            self.max_generations = (budget - pop_size) // pop_size
        else:
            self.max_generations = 0

    def __call__(self, func):
        # Reset seed for reproducibility
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        if pop_size <= 0:
            return (float('inf'), np.full(dim, np.nan))
        # Initial population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.empty(pop_size)
        best_value = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_value:
                best_value = val
                best_x = x.copy()
        # If budget exhausted after initial pop, return
        if evals >= self.budget:
            report_best(best_value, best_x)
            return (best_value, best_x)
        # Main DE loop
        generation = 0
        while evals < self.budget and generation < self.max_generations:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Mutation: select three distinct random indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + 0.5 * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                cross_points = np.random.rand(dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_value:
                        best_value = val
                        best_x = trial.copy()
                        report_best(best_value, best_x)
            generation += 1
        report_best(best_value, best_x)
        return (best_value, best_x)