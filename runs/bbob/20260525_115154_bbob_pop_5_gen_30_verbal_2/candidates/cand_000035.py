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

        # population size: at least 4*dim, cap to budget/2, min 3
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
        CR = 0.9  # initial crossover rate
        F_low, F_high = 0.5, 1.0  # dithering range
        # stagnation detection: restart if no improvement for this many generations
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val

        # main loop
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                # dithering F per individual
                F = rng.uniform(F_low, F_high)
                # mutant using rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                # clip
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover with adaptive CR
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
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # adapt CR based on success (improvement of best this generation)
            if best_val < prev_best_val:
                CR = CR + (1.0 - CR) * 0.1  # increase CR
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                CR = CR * 0.9  # decrease CR
                gen_no_improve += 1
            CR = np.clip(CR, 0.1, 0.9)  # keep in reasonable range

            # restart if stagnation
            if gen_no_improve >= gen_max_restart and evals < budget:
                # restart with diversification: keep best, generate new population
                new_pop = np.empty((pop_size, dim))
                # keep best
                new_pop[0] = best_x
                # fill rest: half uniform, half perturbed best
                n_unif = (pop_size - 1) // 2
                n_pert = pop_size - 1 - n_unif
                new_pop[1:1+n_unif] = rng.uniform(lb, ub, size=(n_unif, dim))
                # perturb best with Gaussian noise, std = (ub-lb)/4, clipped
                std = (ub - lb) / 4.0
                perturbs = best_x + rng.normal(0, std, size=(n_pert, dim))
                perturbs = np.clip(perturbs, lb, ub)
                new_pop[1+n_unif:] = perturbs
                pop = new_pop
                # reevaluate fitness (except best)
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
                CR = 0.9  # reset CR after restart

        return best_val, best_x