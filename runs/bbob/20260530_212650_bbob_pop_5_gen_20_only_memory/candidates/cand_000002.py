import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.evals = 0
        self.best_value = np.inf
        self.best_x = None

    def __call__(self, func):
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Population size: 4*dim, but at most budget/2 to allow generations
        pop_size = min(4 * dim, budget // 2)
        if pop_size < 4:
            pop_size = max(4, min(dim, budget))
        # Ensure at least 3 individuals for DE mutation
        if pop_size < 3:
            pop_size = 3
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if self.evals >= budget:
                break
            x = pop[i]
            val = func(x)
            self.evals += 1
            pop_fitness[i] = val
            if val < self.best_value:
                self.best_value = val
                self.best_x = x.copy()
                report_best(val, x)

        # Main DE loop
        while self.evals < budget:
            for i in range(pop_size):
                if self.evals >= budget:
                    break
                # Choose three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    break
                selected = rng.choice(candidates, 3, replace=False)
                a, b, c = selected
                # Mutation
                F = 0.8
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                CR = 0.9
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Clip
                trial = np.clip(trial, lb, ub)
                # Evaluate
                val = func(trial)
                self.evals += 1
                # Greedy selection
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = trial.copy()
                        report_best(val, trial)

        return self.best_value, self.best_x