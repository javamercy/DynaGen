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

        # Population size heuristic
        pop_size = max(4, min(4 * dim, budget // 2))
        if pop_size > budget:
            pop_size = budget

        # Fallback to random search if population is too small for DE
        if pop_size < 4:
            best_val = np.inf
            best_x = None
            evals = 0
            while evals < budget:
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

        # Stagnation parameters
        max_stag_gen = 20
        stag_count = 0

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
                # Per-individual dither F and CR
                F = rng.uniform(0.5, 1.0)
                CR = rng.uniform(0.7, 1.0)
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
                        report_best(best_val, best_x)
                        improved = True
            if improved:
                stag_count = 0
            else:
                stag_count += 1

            # Restart if stagnation
            if stag_count >= max_stag_gen and evals < budget:
                # Keep best individual
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x.copy()
                fitness[0] = best_val
                # Reinitialize rest
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    new_pop[i] = rng.uniform(lb, ub, size=dim)
                    val = func(new_pop[i])
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_pop[i].copy()
                        report_best(best_val, best_x)
                pop = new_pop
                stag_count = 0

        return best_val, best_x