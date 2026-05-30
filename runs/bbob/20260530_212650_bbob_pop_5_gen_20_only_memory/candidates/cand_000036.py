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

        # Smaller population for exploitation: max(3, 2*dim), capped to budget/4 to allow generations
        pop_size = max(3, min(2 * dim, budget // 4))
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

        # Determine best index in population
        best_idx = np.argmin(pop_fitness)

        # DE/current-to-best/1/bin parameters
        F = 0.5
        CR = 0.95

        while self.evals < budget:
            for i in range(pop_size):
                if self.evals >= budget:
                    break
                # Select two distinct random indices different from i and best_idx
                candidates = [j for j in range(pop_size) if j != i and j != best_idx]
                if len(candidates) < 2:
                    continue
                a, b = rng.choice(candidates, 2, replace=False)
                # Mutation: current-to-best
                mutant = pop[i] + F * (pop[best_idx] - pop[i]) + F * (pop[a] - pop[b])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Clip
                trial = np.clip(trial, lb, ub)
                # Evaluate
                val = func(trial)
                self.evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = trial.copy()
                        report_best(val, trial)
            # Update best index
            best_idx = np.argmin(pop_fitness)

        return self.best_value, self.best_x