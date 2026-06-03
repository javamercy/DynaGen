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

        # population size: at least 4*dim, but cap to budget/2 and at least 3
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # initialize population, F and CR
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        F = rng.uniform(0.1, 0.9, size=pop_size)
        CR = rng.uniform(0, 1, size=pop_size)
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

        # DE parameters
        tau1 = 0.1
        tau2 = 0.1
        stagnation_generations = 0
        max_stagnation = 5 * dim

        generation = 0
        while evals < budget:
            generation += 1
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                # mutation with individual F
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # generate new F and CR for trial
                new_F = F[i]
                new_CR = CR[i]
                if rng.rand() < tau1:
                    new_F = 0.1 + 0.8 * rng.rand()  # [0.1, 0.9] but careful: 0.8*rand gives [0,0.8], plus 0.1 gives [0.1,0.9]
                if rng.rand() < tau2:
                    new_CR = rng.rand()
                # binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                # evaluate trial
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    # update individual
                    fitness[i] = trial_fit
                    pop[i] = trial
                    F[i] = new_F
                    CR[i] = new_CR
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
                # sort population by fitness
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                F = F[order]
                CR = CR[order]
                # keep best 30% (at least 1)
                keep = max(1, int(0.3 * pop_size))
                # reinitialize the rest
                new_pop_size = pop_size - keep
                if new_pop_size > 0:
                    new_pop = rng.uniform(lb, ub, size=(new_pop_size, dim))
                    new_F = rng.uniform(0.1, 0.9, size=new_pop_size)
                    new_CR = rng.uniform(0, 1, size=new_pop_size)
                    # evaluate new individuals
                    for i in range(new_pop_size):
                        if evals >= budget:
                            break
                        new_fit = func(new_pop[i])
                        evals += 1
                        if new_fit < fitness[keep + i]:
                            fitness[keep + i] = new_fit
                            pop[keep + i] = new_pop[i]
                            F[keep + i] = new_F[i]
                            CR[keep + i] = new_CR[i]
                            if new_fit < best_val:
                                best_val = new_fit
                                best_x = new_pop[i].copy()
                                report_best(best_val, best_x)
                stagnation_generations = 0

        return best_val, best_x