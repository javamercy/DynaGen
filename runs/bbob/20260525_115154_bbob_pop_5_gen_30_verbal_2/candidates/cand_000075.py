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
        if pop_size < 3:
            pop_size = 3

        # initial population
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

        # JADE parameters
        mean_F = 0.5
        mean_CR = 0.5
        c = 0.1
        p_best = 0.1

        # stagnation
        max_stag = max(1, budget // (2 * pop_size))
        stag_gen = 0
        prev_best = best_val

        while evals < budget:
            # generate F and CR
            F = np.clip(rng.standard_cauchy(pop_size) * 0.1 + mean_F, 0, 1)
            CR = np.clip(rng.standard_cauchy(pop_size) * 0.1 + mean_CR, 0, 1)
            # ensure variation
            if np.std(F) < 1e-10:
                F += rng.uniform(-0.1, 0.1, pop_size)
                F = np.clip(F, 0, 1)
            if np.std(CR) < 1e-10:
                CR += rng.uniform(-0.1, 0.1, pop_size)
                CR = np.clip(CR, 0, 1)

            # select pbest indices
            sorted_idx = np.argsort(fitness)
            n_pbest = max(1, int(p_best * pop_size))
            pbest_idx = sorted_idx[:n_pbest]

            succ_F = []
            succ_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                # mutation: current-to-pbest/1 (no archive)
                pbest_i = pbest_idx[rng.randint(n_pbest)]
                # select two distinct indices from population excluding i
                indices = [j for j in range(pop_size) if j != i]
                if len(indices) < 2:
                    continue
                r1, r2 = rng.choice(indices, size=2, replace=False)
                mutant = pop[i] + F[i] * (pop[pbest_i] - pop[i]) + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)

                # crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    # update population
                    fitness[i] = trial_fit
                    pop[i] = trial
                    succ_F.append(F[i])
                    succ_CR.append(CR[i])
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update mean_F and mean_CR
            if len(succ_F) > 0:
                mean_F = (1 - c) * mean_F + c * (np.sum(np.square(succ_F)) / np.sum(succ_F))
                mean_CR = (1 - c) * mean_CR + c * np.mean(succ_CR)

            # stagnation check
            if best_val < prev_best:
                stag_gen = 0
                prev_best = best_val
            else:
                stag_gen += 1

            if stag_gen >= max_stag and evals < budget:
                # restart: keep best, fill rest with Gaussian noise scaled by component-wise std
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                # compute component-wise std of current population
                std_pop = np.std(pop, axis=0)
                # avoid zero std
                std_pop = np.maximum(std_pop, 1e-10 * (ub - lb))
                # scaling factor
                scale = 0.2
                for i in range(1, pop_size):
                    # Gaussian perturbation
                    new_pop[i] = best_x + rng.normal(0, std_pop * scale, dim)
                    new_pop[i] = np.clip(new_pop[i], lb, ub)
                pop = new_pop
                # evaluate new population (except best)
                fitness = np.full(pop_size, np.inf)
                fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                # reset stagnation
                stag_gen = 0
                prev_best = best_val

        return best_val, best_x