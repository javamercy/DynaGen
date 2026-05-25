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
        rng = self.rng
        budget = self.budget

        # Population size: between 4 and min(10*dim, budget//4)
        pop_size = max(4, min(10 * dim, budget // 4))
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # Main DE loop
        F = 0.5
        CR = 0.9
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Choose three distinct random indices different from i
                indices = list(range(pop_size))
                indices.remove(i)
                rng.shuffle(indices)
                a, b, c = indices[:3]
                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                j_rand = rng.randint(dim)
                trial = np.copy(pop[i])
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                # Selection
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