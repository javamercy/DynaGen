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

        # archive
        archive = []
        archive_max = pop_size

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
            # ensure some variation
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
                # choose pbest randomly
                pbest_i = pbest_idx[rng.randint(n_pbest)]
                # union of pop and archive indices (excluding i)
                union_indices = list(range(pop_size)) + [pop_size + j for j in range(len(archive))]
                union_indices = [idx for idx in union_indices if idx != i]
                # sample two distinct indices
                if len(union_indices) < 2:
                    continue
                r1, r2 = rng.choice(union_indices, size=2, replace=False)
                # retrieve vectors
                if r1 < pop_size:
                    x1 = pop[r1]
                else:
                    x1 = archive[r1 - pop_size]
                if r2 < pop_size:
                    x2 = pop[r2]
                else:
                    x2 = archive[r2 - pop_size]
                mutant = pop[i] + F[i] * (pop[pbest_i] - pop[i]) + F[i] * (x1 - x2)
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
                    # add parent to archive
                    if len(archive) < archive_max:
                        archive.append(pop[i].copy())
                    else:
                        idx = rng.randint(archive_max)
                        archive[idx] = pop[i].copy()
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
                # restart: keep best, fill rest with mixture
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                # population spread for scaling
                std_pop = np.std(pop, axis=0)
                scale_fixed = (ub - lb) * 0.2
                scale_small = (ub - lb) * 0.05
                for i in range(1, pop_size):
                    if rng.rand() < 0.2:
                        # uniform
                        new_pop[i] = rng.uniform(lb, ub, dim)
                    else:
                        # large Cauchy perturbation (40%) or small (40%)
                        if rng.rand() < 0.5:
                            scale = np.maximum(scale_fixed, 0.1 * std_pop)
                        else:
                            scale = np.maximum(scale_small, 0.02 * std_pop)
                        new_pop[i] = best_x + rng.standard_cauchy(dim) * scale
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
                # clear archive and reset stagnation
                archive.clear()
                stag_gen = 0
                prev_best = best_val

        return best_val, best_x