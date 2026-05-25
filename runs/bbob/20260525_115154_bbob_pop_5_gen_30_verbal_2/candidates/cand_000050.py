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

        # population size: larger for exploration
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
        CRm = 0.9  # mean CR for Cauchy sampling
        CR_std = 0.1
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val
        successful_CR = []

        generation = 0
        while evals < budget:
            generation += 1
            # generate CR for each individual
            CR = np.clip(rng.standard_cauchy(pop_size) * CR_std + CRm, 0, 1)
            if np.std(CR) < 1e-10:
                CR += rng.uniform(-0.1, 0.1, pop_size)
                CR = np.clip(CR, 0, 1)

            for i in range(pop_size):
                if evals >= budget:
                    break
                # mutation
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                F = rng.uniform(0.5, 1.0)
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

            # update CRm
            if len(successful_CR) > 0:
                CRm = (1 - 0.1) * CRm + 0.1 * np.mean(successful_CR)
                successful_CR = []

            # stagnation check
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            # diversification: replace some individuals with random points
            for i in range(pop_size):
                if evals >= budget:
                    break
                if rng.rand() < 0.05:
                    new_point = rng.uniform(lb, ub, dim)
                    new_fit = func(new_point)
                    evals += 1
                    if new_fit < fitness[i]:
                        fitness[i] = new_fit
                        pop[i] = new_point
                        if new_fit < best_val:
                            best_val = new_fit
                            best_x = new_point.copy()
                            report_best(best_val, best_x)

            # restart if stagnation
            if gen_no_improve >= gen_max_restart and evals < budget:
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                scale = (ub - lb) * 0.5  # 50% of domain
                for i in range(1, pop_size):
                    if rng.rand() < 0.5:
                        new_pop[i] = best_x + rng.standard_cauchy(dim) * scale
                    else:
                        new_pop[i] = rng.uniform(lb, ub, dim)
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