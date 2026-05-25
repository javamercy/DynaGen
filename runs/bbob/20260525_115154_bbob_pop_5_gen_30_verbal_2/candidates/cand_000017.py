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
        span = ub - lb

        # population size
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # initialize
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
        F = 0.5
        CR = 0.9
        stagnation_threshold = 3 * dim
        stagnation = 0

        # Main DE loop
        while evals < budget:
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct random indices
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = rng.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # crossover
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
                stagnation = 0
            else:
                stagnation += 1
            # restart if stagnation
            if stagnation >= stagnation_threshold and evals < budget:
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                keep = max(1, int(0.3 * pop_size))
                # reinitialize the rest around best with local Gaussian
                sigma = 0.2 * span
                new_pop = rng.normal(loc=pop[0], scale=sigma, size=(pop_size - keep, dim))
                new_pop = np.clip(new_pop, lb, ub)
                for i in range(pop_size - keep):
                    if evals >= budget:
                        break
                    new_fit = func(new_pop[i])
                    evals += 1
                    if new_fit < best_val:
                        best_val = new_fit
                        best_x = new_pop[i].copy()
                        report_best(best_val, best_x)
                    pop[keep + i] = new_pop[i]
                    fitness[keep + i] = new_fit
                stagnation = 0

        # Final local search (if budget remains)
        if evals < budget:
            for _ in range(budget - evals):
                step = 0.05 * span * (1 - evals / budget)
                candidate = best_x + rng.normal(0, step, dim)
                candidate = np.clip(candidate, lb, ub)
                cand_val = func(candidate)
                evals += 1
                if cand_val < best_val:
                    best_val = cand_val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x