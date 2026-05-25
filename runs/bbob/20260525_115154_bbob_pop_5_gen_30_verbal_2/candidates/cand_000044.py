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

        # population size
        pop_size = max(4 * dim, 3)
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

        # success-history memories
        H = 10
        success_F = [0.8] * H
        success_CR = [0.9] * H
        F_idx = 0
        CR_idx = 0

        stagnation = 0
        max_stagnation = max(5 * dim, 50)

        while evals < budget:
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # sample F and CR from success history
                F = rng.choice(success_F)
                CR = rng.choice(success_CR)
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
                # evaluate
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    # update success memories
                    success_F[F_idx % H] = F
                    success_CR[CR_idx % H] = CR
                    F_idx += 1
                    CR_idx += 1
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    improved = True

            # local search on best
            if evals < budget:
                sigma = 0.05 * range_
                trial = best_x + sigma * rng.randn(dim)
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                if trial_fit < best_val:
                    best_val = trial_fit
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    improved = True

            if not improved:
                stagnation += 1
            else:
                stagnation = 0

            # restart if stagnation
            if stagnation >= max_stagnation and evals < budget:
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                keep = max(1, int(0.3 * pop_size))
                new_pop_size = pop_size - keep
                if new_pop_size > 0:
                    # half uniform, half Gaussian around best
                    half = new_pop_size // 2
                    # uniform
                    if half > 0:
                        new_pop_uniform = rng.uniform(lb, ub, size=(half, dim))
                        for i in range(half):
                            if evals >= budget:
                                break
                            new_fit = func(new_pop_uniform[i])
                            evals += 1
                            idx = keep + i
                            if new_fit < fitness[idx]:
                                fitness[idx] = new_fit
                                pop[idx] = new_pop_uniform[i]
                                if new_fit < best_val:
                                    best_val = new_fit
                                    best_x = new_pop_uniform[i].copy()
                                    report_best(best_val, best_x)
                    # Gaussian around best kept (pop[0])
                    gaussian_half = new_pop_size - half
                    if gaussian_half > 0:
                        sigma = 0.2 * range_
                        mean = pop[0]
                        new_pop_gauss = rng.randn(gaussian_half, dim) * sigma + mean
                        new_pop_gauss = np.clip(new_pop_gauss, lb, ub)
                        for i in range(gaussian_half):
                            if evals >= budget:
                                break
                            new_fit = func(new_pop_gauss[i])
                            evals += 1
                            idx = keep + half + i
                            if new_fit < fitness[idx]:
                                fitness[idx] = new_fit
                                pop[idx] = new_pop_gauss[i]
                                if new_fit < best_val:
                                    best_val = new_fit
                                    best_x = new_pop_gauss[i].copy()
                                    report_best(best_val, best_x)
                stagnation = 0

        return best_val, best_x