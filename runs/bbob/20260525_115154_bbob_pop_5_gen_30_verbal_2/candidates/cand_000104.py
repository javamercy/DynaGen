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
        range_len = ub - lb

        # population size scaling
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
        mu_F = 0.5
        mu_CR = 0.5
        archive = []  # list of inferior vectors
        max_archive_size = pop_size

        # stagnation tracking
        gen_no_improve = 0
        prev_best_val = best_val
        gen_max_restart = max(1, budget // (2 * pop_size))

        while evals < budget:
            # generate F and CR for each target
            F_i = np.clip(mu_F + 0.1 * rng.standard_cauchy(pop_size), 0, 1)
            CR_i = np.clip(mu_CR + 0.1 * rng.randn(pop_size), 0, 1)
            # ensure at least one F and CR valid
            F_i = np.clip(F_i, 0.1, 0.9)
            CR_i = np.clip(CR_i, 0.1, 0.9)

            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # select a,b,c distinct and different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                # optionally use archive for a or b
                # simple rand/1 without archive influence on selection
                a, b, c = rng.choice(candidates, size=3, replace=False)

                F = F_i[i]
                CR = CR_i[i]

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
                    # archive the replaced vector
                    if len(archive) < max_archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # replace random archive element
                        idx = rng.randint(max_archive_size)
                        archive[idx] = pop[i].copy()
                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F)
                    successful_CR.append(CR)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update mu_F and mu_CR using successful parameters
            if len(successful_F) > 0:
                mu_F = (1 - 0.1) * mu_F + 0.1 * np.sum(np.array(successful_F)**2) / np.sum(np.array(successful_F))
                mu_CR = (1 - 0.1) * mu_CR + 0.1 * np.mean(successful_CR)

            # check stagnation
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= gen_max_restart and evals < budget:
                # restart: keep best, generate rest around best with Cauchy perturbation
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x.copy()
                for i in range(1, pop_size):
                    # perturbation: best + 0.2 * range * Cauchy(0,1)
                    perturb = 0.2 * range_len * rng.standard_cauchy(dim)
                    x_try = best_x + perturb
                    x_try = np.clip(x_try, lb, ub)
                    new_pop[i] = x_try
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
                # reset parameters
                mu_F = 0.5
                mu_CR = 0.5
                archive.clear()
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x