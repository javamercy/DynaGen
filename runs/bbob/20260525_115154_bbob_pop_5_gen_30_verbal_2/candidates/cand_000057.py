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
        range_ = ub - lb

        # population size
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # initialize population
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

        stagnation_generations = 0
        max_stagnation = 10 * dim  # increased threshold
        generation = 0

        while evals < budget:
            generation += 1
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # fixed F and CR
                F = 0.8
                CR = 0.9
                # select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # evaluate
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

            if not improved:
                stagnation_generations += 1
            else:
                stagnation_generations = 0

            # restart if stagnation
            if stagnation_generations >= max_stagnation and evals < budget:
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                keep = max(1, int(0.3 * pop_size))
                new_pop_size = pop_size - keep
                if new_pop_size > 0:
                    new_pop = rng.uniform(lb, ub, size=(new_pop_size, dim))
                    for i in range(new_pop_size):
                        if evals >= budget:
                            break
                        new_fit = func(new_pop[i])
                        evals += 1
                        if new_fit < fitness[keep + i]:
                            fitness[keep + i] = new_fit
                            pop[keep + i] = new_pop[i]
                            if new_fit < best_val:
                                best_val = new_fit
                                best_x = new_pop[i].copy()
                                report_best(best_val, best_x)
                stagnation_generations = 0

        return best_val, best_x