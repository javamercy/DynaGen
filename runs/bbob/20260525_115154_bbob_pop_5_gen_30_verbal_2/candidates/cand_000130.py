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

        # DE parameters
        CRm = 0.9
        CR_std = 0.1
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val
        successful_CR = []
        generation = 0
        max_generations = budget // pop_size  # approximate max generations

        while evals < budget:
            generation += 1
            # schedule parameters
            t = generation / max_generations if max_generations > 0 else 1.0
            p_cb = min(0.9, 0.3 + 0.6 * t)  # probability of current-to-best
            F_low = max(0.3, 0.5 - 0.2 * t)
            F_high = min(0.9, 1.0 - 0.3 * t)
            # restart proportions
            p_uniform = max(0.1, 0.3 - 0.2 * t)  # from 0.3 to 0.1
            p_large = 0.5 - p_uniform / 2  # so sum to 1
            p_small = 1 - p_uniform - p_large
            # generate CR for each individual via Cauchy
            CR = np.clip(rng.standard_cauchy(pop_size) * CR_std + CRm, 0, 1)
            if np.std(CR) < 1e-10:
                CR += rng.uniform(-0.1, 0.1, pop_size)
                CR = np.clip(CR, 0, 1)

            for i in range(pop_size):
                if evals >= budget:
                    break
                # choose mutation strategy
                if rng.rand() < p_cb:
                    # DE/current-to-best/1
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    a, b = rng.choice(candidates, size=2, replace=False)
                    F = rng.uniform(F_low, F_high)
                    mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                else:
                    # DE/rand/1
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    a, b, c = rng.choice(candidates, size=3, replace=False)
                    F = rng.uniform(F_low, F_high)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                # evaluation
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_CR.append(CR[i])
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update CRm via Lehmer mean
            if len(successful_CR) > 0:
                CRm = (1 - 0.1) * CRm + 0.1 * (np.sum(np.square(successful_CR)) / np.sum(successful_CR))
                successful_CR = []

            # stagnation check
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= gen_max_restart and evals < budget:
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                scale_large = (ub - lb) * 0.2
                scale_small = (ub - lb) * 0.05
                for i in range(1, pop_size):
                    rand_val = rng.rand()
                    if rand_val < p_uniform:
                        # uniform sampling
                        new_pop[i] = rng.uniform(lb, ub, size=dim)
                    elif rand_val < p_uniform + p_large:
                        # large Cauchy perturbation
                        new_pop[i] = best_x + rng.standard_cauchy(dim) * scale_large
                    else:
                        # small Cauchy perturbation
                        new_pop[i] = best_x + rng.standard_cauchy(dim) * scale_small
                    new_pop[i] = np.clip(new_pop[i], lb, ub)
                pop = new_pop
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
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x