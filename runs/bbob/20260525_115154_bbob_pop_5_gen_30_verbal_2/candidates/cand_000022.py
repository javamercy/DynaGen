import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Determine population size
        pop_size = max(4, min(4 * dim, budget // 2))
        if pop_size > budget:
            pop_size = budget

        # Fallback to random search if population is too small for DE/rand/1/bin
        if pop_size < 4:
            best_val = np.inf
            best_x = None
            evals = 0
            for _ in range(budget):
                x = rng.uniform(lb, ub, size=dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # DE parameters
        F = 0.8
        CR = 0.9
        stagnation_limit = 20
        stagnation_counter = 0

        # Main DE loop
        while evals < budget:
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                # Mutant
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        improved = True
                        report_best(best_val, best_x)

            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= stagnation_limit:
                # Restart: keep the best, reinitialize the rest randomly
                pop[0] = best_x.copy()
                fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    pop[i] = rng.uniform(lb, ub, size=dim)
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                stagnation_counter = 0

        return best_val, best_x