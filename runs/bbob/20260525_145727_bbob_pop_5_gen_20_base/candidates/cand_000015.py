import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Population size: at least 4, at most 10*dim, and at most half the budget
        self.pop_size = max(4, min(10*dim, budget // 2))
        # Ensure pop_size <= budget
        if self.pop_size > budget:
            self.pop_size = budget
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        F = self.F
        CR = self.CR
        rng = self.rng

        # Initialize population uniformly within bounds
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # Evaluate initial population
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

        # Main DE loop
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Generate three distinct random indices not equal to i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover: binomial
                jrand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == jrand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i, j]
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                # Evaluate trial
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x