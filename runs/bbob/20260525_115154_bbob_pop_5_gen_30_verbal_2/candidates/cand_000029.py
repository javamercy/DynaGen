import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_ = ub - lb

        # population size
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)

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

        # DE parameters memory (success-history)
        H = 6  # memory size
        F_memory = np.full(H, 0.5)
        CR_memory = np.full(H, 0.5)
        mem_idx = 0

        stagnation_generations = 0
        max_stagnation = 5 * dim

        generation = 0
        while evals < budget:
            generation += 1
            improved = False
            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # generate F and CR using adaptive sampling
                # choose random memory index
                k = rng.randint(H)
                # sample F from Cauchy centered at F_memory[k]
                F = F_memory[k] + 0.1 * rng.standard_cauchy()
                F = np.clip(F, 0.0, 1.0)
                # sample CR from Normal centered at CR_memory[k]
                CR = CR_memory[k] + 0.1 * rng.randn()
                CR = np.clip(CR, 0.0, 1.0)

                # select three distinct indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids

                # mutation and crossover
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

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
                    successful_F.append(F)
                    successful_CR.append(CR)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved = True

            # update memory with successful parameters
            if len(successful_F) > 0:
                # arithmetic mean
                F_mean = np.mean(successful_F)
                CR_mean = np.mean(successful_CR)
                F_memory[mem_idx] = F_mean
                CR_memory[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % H

            if not improved:
                stagnation_generations += 1
            else:
                stagnation_generations = 0

            # restart on stagnation
            if stagnation_generations >= max_stagnation and evals < budget:
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                keep = max(1, int(0.3 * pop_size))
                new_size = pop_size - keep
                if new_size > 0:
                    new_pop = rng.uniform(lb, ub, size=(new_size, dim))
                    for i in range(new_size):
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