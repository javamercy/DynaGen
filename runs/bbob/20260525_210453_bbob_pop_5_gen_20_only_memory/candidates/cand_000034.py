import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        # population size, at least 4 for DE/rand/1 mutation
        pop_size = min(5 * dim, max(10, budget // 4))
        pop_size = max(4, min(pop_size, budget // 2))  # ensure at least 4 and not too large
        pop_size = min(pop_size, budget)  # cannot exceed budget
        if pop_size < 4:
            # budget too small, simple random search
            best_x = None
            best_f = np.inf
            fevals = 0
            while fevals < budget:
                x = np.random.uniform(lb, ub)
                val = func(x)
                fevals += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                    report_best(best_f, best_x)
            return best_f, best_x

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fevals = 0

        # initial evaluation
        for i in range(pop_size):
            if fevals >= budget:
                break
            val = func(pop[i])
            fevals += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = pop[i].copy()
                report_best(best_f, best_x)

        # DE parameters
        F = 0.8
        CR = 0.9
        max_stag_gen = max(5, dim // 10)  # stagnation threshold
        no_improve_gen = 0

        while fevals < budget:
            # mutation, crossover, selection for each individual
            for i in range(pop_size):
                if fevals >= budget:
                    break
                # select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = random.sample(candidates, 3)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                val = func(trial)
                fevals += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        no_improve_gen = 0
                    else:
                        no_improve_gen += 1
                else:
                    no_improve_gen += 1

            # check stagnation
            if no_improve_gen >= max_stag_gen * pop_size:
                # restart worst half of population
                n_restart = pop_size // 2
                sorted_indices = np.argsort(pop_f)
                worst_indices = sorted_indices[-n_restart:]
                for idx in worst_indices:
                    pop[idx] = np.random.uniform(lb, ub)
                    # evaluate new points (optional: could save budget, but we evaluate them later)
                    # To avoid extra evaluations, we can evaluate them now or in next iteration.
                    # We'll evaluate them now to update best
                    if fevals < budget:
                        val = func(pop[idx])
                        fevals += 1
                        pop_f[idx] = val
                        if val < best_f:
                            best_f = val
                            best_x = pop[idx].copy()
                            report_best(best_f, best_x)
                no_improve_gen = 0
            # else continue

        return best_f, best_x