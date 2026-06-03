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

        # population size
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)

        # store F and CR for each individual
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.5)

        # initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # parameters for adaptation
        tau1 = 0.1
        tau2 = 0.1
        F_low = 0.1
        F_high = 0.9
        CR_low = 0.0
        CR_high = 1.0

        # stagnation detection
        stagnation_limit = min(50, max(10, int(0.1 * budget / pop_size)))
        no_improve_gen = 0

        # main loop
        while evals < budget:
            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                # choose three distinct random indices not i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                # mutation
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
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
                # update F and CR
                if rng.rand() < tau1:
                    F[i] = F_low + rng.rand() * (F_high - F_low)
                if rng.rand() < tau2:
                    CR[i] = CR_low + rng.rand() * (CR_high - CR_low)

            # check for improvement
            if evals >= budget:
                break
            new_best_val = np.min(fitness)
            if new_best_val < best_val - 1e-12:
                best_val = new_best_val
                # best_x already updated
                no_improve_gen = 0
            else:
                no_improve_gen += 1

            # restart if stagnant
            if no_improve_gen >= stagnation_limit:
                # keep best individual
                best_idx = np.argmin(fitness)
                best_individual = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                # reinitialize others
                for i in range(pop_size):
                    if i == best_idx:
                        continue
                    if evals >= budget:
                        break
                    pop[i] = rng.uniform(lb, ub, size=dim)
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                    # reset F and CR
                    F[i] = 0.5 + 0.4 * rng.rand()
                    CR[i] = 0.5 * rng.rand()
                # ensure best is still there after possible overwrite
                pop[best_idx] = best_individual
                fitness[best_idx] = best_fit
                no_improve_gen = 0

        return best_val, best_x