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

        # population size: at least 4*dim, but cap to budget/2 and at least 3
        pop_size = max(4*dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
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
        F = 0.8
        CR = 0.9
        stagnant_gen = 0
        prev_best_val = best_val

        while evals < budget:
            success_count = 0
            total_count = 0

            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                # mutant
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # evaluate
                trial_fit = func(trial)
                evals += 1
                total_count += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    success_count += 1
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # adapt F and CR based on success rate
            if total_count > 0:
                success_rate = success_count / total_count
                if success_rate < 0.2:
                    CR = min(1.0, CR + 0.05)
                    F = min(1.0, F + 0.05)
                elif success_rate > 0.5:
                    CR = max(0.1, CR - 0.05)
                    F = max(0.1, F - 0.05)

            # check stagnation
            if best_val < prev_best_val:
                stagnant_gen = 0
                prev_best_val = best_val
            else:
                stagnant_gen += 1

            # diversity-based restart if stalled
            if stagnant_gen >= 5 and evals < budget:
                # compute mean pairwise distance
                center = np.mean(pop, axis=0)
                dists = np.linalg.norm(pop - center, axis=1)
                mean_dist = np.mean(dists)
                range_norm = np.linalg.norm(ub - lb)
                if mean_dist < 0.01 * range_norm:
                    # reinitialize 20% of population (excluding best)
                    num_reinit = max(1, int(0.2 * pop_size))
                    # find indices excluding best
                    best_idx = np.argmin(fitness)
                    all_indices = list(range(pop_size))
                    all_indices.remove(best_idx)
                    if len(all_indices) >= num_reinit:
                        reinit_indices = rng.choice(all_indices, size=num_reinit, replace=False)
                        for idx in reinit_indices:
                            if evals >= budget:
                                break
                            new_ind = rng.uniform(lb, ub, size=dim)
                            new_fit = func(new_ind)
                            evals += 1
                            if new_fit < fitness[idx]:
                                fitness[idx] = new_fit
                                pop[idx] = new_ind
                                if new_fit < best_val:
                                    best_val = new_fit
                                    best_x = new_ind.copy()
                                    report_best(best_val, best_x)
                    stagnant_gen = 0  # reset after restart

        return best_val, best_x