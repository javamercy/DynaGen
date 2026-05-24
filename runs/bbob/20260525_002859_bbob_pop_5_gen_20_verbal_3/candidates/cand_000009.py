import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0

        # population size
        popsize = min(budget, max(4, min(5*dim, 20)))
        if popsize < 4:
            popsize = 4
        if popsize > budget:
            popsize = budget

        # initialize population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fitness = np.full(popsize, np.inf)
        best_val = np.inf
        best_x = None

        for i in range(popsize):
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # adaptive parameters
        F = 0.5
        CR = 0.9
        F_archive = []
        CR_archive = []
        num_success = 0

        while evals < budget:
            # generate new F and CR from archives or defaults
            if len(F_archive) > 0:
                F = rng.choice(F_archive)
            else:
                F = 0.5 + 0.1 * rng.randn()
            if len(CR_archive) > 0:
                CR = rng.choice(CR_archive)
            else:
                CR = 0.9 + 0.1 * rng.randn()
            F = np.clip(F, 0.1, 0.9)
            CR = np.clip(CR, 0.0, 1.0)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            for i in range(popsize):
                # select distinct indices
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                # mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                # crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                # evaluate
                val = func(trial)
                evals += 1
                if val <= new_fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    # record successful parameters
                    F_archive.append(F)
                    CR_archive.append(CR)
                    num_success += 1
                    # limit archive size
                    if len(F_archive) > popsize:
                        F_archive.pop(0)
                        CR_archive.pop(0)
                if evals >= budget:
                    break
            if evals >= budget:
                break
            pop = new_pop
            fitness = new_fitness
            # update F and CR means based on success rate
            if num_success > 0:
                F = np.mean(F_archive)
                CR = np.mean(CR_archive)

        return best_val, best_x