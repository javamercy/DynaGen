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

        # small population size
        pop_size = max(3, min(4*dim, budget // 10))
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

        # DE parameters
        CR = 0.9
        # stagnation detection
        no_improve_gen = 0
        max_no_improve = max(10, budget // (pop_size * 20))

        while evals < budget:
            # dithering F per generation
            F = 0.5 + rng.rand() * 0.5
            for i in range(pop_size):
                if evals >= budget:
                    break
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
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        no_improve_gen = 0
                    else:
                        no_improve_gen += 1
                else:
                    no_improve_gen += 1
            # restart if stagnation
            if no_improve_gen > max_no_improve and evals < budget:
                # keep best, reinitialize rest
                new_pop = [best_x] if best_x is not None else []
                while len(new_pop) < pop_size:
                    new_pop.append(rng.uniform(lb, ub, size=dim))
                pop = np.array(new_pop)
                # reevaluate fitness for new individuals (skip best if kept)
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    if i == 0 and best_x is not None:
                        continue
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                no_improve_gen = 0

        return best_val, best_x