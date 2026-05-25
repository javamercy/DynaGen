import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # population size adaptive
        pop_size = max(4 * dim, 20)
        pop_size = min(pop_size, budget // 2)  # ensure at least one generation
        if pop_size < 1:
            pop_size = 1
        self.pop_size = pop_size
        self.max_generations = (budget - pop_size) // pop_size if pop_size > 0 else 0
        # local search parameters
        self.local_steps_per_gen = max(1, int(0.1 * pop_size))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        if pop_size <= 0:
            return (float('inf'), np.full(dim, np.nan))
        
        # initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
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
                report_best(best_value, best_x)
        
        # main DE loop with local search
        generation = 0
        while evals < self.budget and generation < self.max_generations:
            # DE generation
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # mutation: rand/1
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + 0.5 * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
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
            # local search on best solution
            if best_x is not None:
                step_size = 0.1 * (ub - lb) * (1.0 - evals / self.budget)
                for _ in range(self.local_steps_per_gen):
                    if evals >= self.budget:
                        break
                    perturbation = np.random.normal(0, step_size, dim)
                    candidate = best_x + perturbation
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    evals += 1
                    if val < best_value:
                        best_value = val
                        best_x = candidate.copy()
                        report_best(best_value, best_x)
            generation += 1
        
        return (best_value, best_x)