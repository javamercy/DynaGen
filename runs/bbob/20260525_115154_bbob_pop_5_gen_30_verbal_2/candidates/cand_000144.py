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

        # population size
        pop_size = min(max(4 * dim, 3), budget // 3)
        if pop_size < 3:
            pop_size = 3

        # archive size = pop_size
        archive_size = pop_size

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

        # JADE parameters
        mean_F = 0.5
        mean_CR = 0.5
        c = 0.1
        p_best = 0.1

        # archive
        archive = []  # list of arrays

        # stagnation
        max_stag = max(1, budget // (2 * pop_size))
        stag_gen = 0
        prev_best = best_val

        # diversity threshold
        div_threshold = 0.05 * np.mean(ub - lb)

        while evals < budget:
            # generate F and CR
            F = np.clip(rng.standard_cauchy(pop_size) * 0.1 + mean_F, 0, 1)
            CR = np.clip(rng.standard_cauchy(pop_size) * 0.1 + mean_CR, 0, 1)
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
                # mutation: current-to-pbest/1 with archive
                pbest_i = pbest_idx[rng.randint(n_pbest)]
                # choose r1 from pop (excluding i)
                r1 = rng.randint(pop_size)
                while r1 == i:
                    r1 = rng.randint(pop_size)
                # choose r2 from archive (if any) else pop
                if len(archive) > 0:
                    r2 = rng.randint(len(archive))
                    r2_point = archive[r2]
                else:
                    r2 = rng.randint(pop_size)
                    while r2 == i or r2 == r1:
                        r2 = rng.randint(pop_size)
                    r2_point = pop[r2]
                mutant = pop[i] + F[i] * (pop[pbest_i] - pop[i]) + F[i] * (pop[r1] - r2_point)
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    # update population and archive
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(0)
                    fitness[i] = trial_fit
                    pop[i] = trial
                    succ_F.append(F[i])
                    succ_CR.append(CR[i])
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update means
            if len(succ_F) > 0:
                mean_F = (1 - c) * mean_F + c * (np.sum(np.square(succ_F)) / np.sum(succ_F))
                mean_CR = (1 - c) * mean_CR + c * np.mean(succ_CR)

            # stagnation check
            if best_val < prev_best:
                stag_gen = 0
                prev_best = best_val
            else:
                stag_gen += 1

            # diversity check (average distance to best)
            if best_x is not None and len(pop) > 0:
                dists = np.mean(np.abs(pop - best_x), axis=1)
                avg_div = np.mean(dists)
            else:
                avg_div = 0

            restart = False
            if stag_gen >= max_stag and evals < budget:
                restart = True
            if avg_div < div_threshold and evals > 0.2 * budget:  # only check after some evals
                restart = True

            if restart:
                # restart: keep best, reinitialize rest
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                for i in range(1, pop_size):
                    if rng.rand() < 0.5:
                        new_pop[i] = rng.uniform(lb, ub, dim)
                    else:
                        std = 0.2 * (ub - lb)
                        new_pop[i] = best_x + rng.normal(0, std, dim)
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
                # reset adaptation and archive
                mean_F = 0.5
                mean_CR = 0.5
                archive.clear()
                stag_gen = 0
                prev_best = best_val

        return best_val, best_x