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
        range_width = ub - lb

        # population size: at least 4*dim, but cap to budget/2 and at least 3
        pop_size = max(4*dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # initialize population uniformly
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = np.empty(dim)
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            fit = func(pop[i])
            evals += 1
            fitness[i] = fit
            if fit < best_val:
                best_val = fit
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # archive for JADE
        archive = []
        archive_size = pop_size

        # adaptive parameters
        mu_F = 0.5
        mu_CR = 0.5
        p_best = 0.1  # fraction of best individuals for pbest mutation

        # stagnation detection
        stagnation_counter = 0
        stagnation_limit = pop_size * 5  # generations without improvement trigger restart

        # main loop
        gen = 0
        while evals < budget:
            # success lists for adaptation
            success_F = []
            success_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # generate F and CR for this individual
                F = rng.cauchy(mu_F, 0.1)
                while F <= 0:
                    F = rng.cauchy(mu_F, 0.1)
                if F > 1:
                    F = 1
                CR = rng.normal(mu_CR, 0.1)
                CR = np.clip(CR, 0, 1)

                # select pbest index
                best_indices = np.argsort(fitness)[:max(1, int(p_best * pop_size))]
                pbest_idx = rng.choice(best_indices)
                pbest = pop[pbest_idx]

                # select two distinct indices from pop + archive (excluding i)
                pop_indices = list(range(pop_size))
                pop_indices.remove(i)
                candidates = pop_indices + archive
                ids = rng.choice(candidates, size=2, replace=False)
                r1, r2 = ids
                if r1 >= pop_size:
                    r1 = archive[r1 - pop_size]
                else:
                    r1 = pop[r1]
                if r2 >= pop_size:
                    r2 = archive[r2 - pop_size]
                else:
                    r2 = pop[r2]

                # current-to-pbest mutation
                mutant = pop[i] + F * (pbest - pop[i]) + F * (r1 - r2)
                # clip to bounds
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # evaluate trial
                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    # success: update pop, archive, and success lists
                    success_F.append(F)
                    success_CR.append(CR)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    # add trial to archive? Not typical, but can help diversity
                    archive.append(trial.copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))
                    stagnation_counter += 1

            # update means based on successes
            if success_F:
                # Lehmer mean for F
                sum_sq = sum(f**2 for f in success_F)
                sum_f = sum(success_F)
                if sum_f > 0:
                    mu_F = 0.9 * mu_F + 0.1 * (sum_sq / sum_f)
                else:
                    mu_F = 0.9 * mu_F + 0.1 * 0.5
                # arithmetic mean for CR
                mu_CR = 0.9 * mu_CR + 0.1 * np.mean(success_CR)
            else:
                mu_F = 0.9 * mu_F + 0.1 * 0.5
                mu_CR = 0.9 * mu_CR + 0.1 * 0.5

            # stagnation restart
            if stagnation_counter >= stagnation_limit and evals < budget:
                # keep best
                best_idx = np.argmin(fitness)
                best_individual = pop[best_idx].copy()
                best_fitness = fitness[best_idx]
                # reinitialize 50% of population (excluding best)
                num_reinit = pop_size // 2
                # ensure we have enough remaining budget
                if evals + num_reinit > budget:
                    num_reinit = budget - evals
                reinit_indices = rng.choice([i for i in range(pop_size) if i != best_idx], size=num_reinit, replace=False)
                for idx in reinit_indices:
                    if evals >= budget:
                        break
                    # uniform random point
                    new_point = rng.uniform(lb, ub, size=dim)
                    # also generate perturbed best
                    if rng.rand() < 0.5:
                        # perturbation scaled by 10% of range
                        new_point = best_x + rng.normal(0, 0.1*range_width, size=dim)
                        new_point = np.clip(new_point, lb, ub)
                    new_fit = func(new_point)
                    evals += 1
                    pop[idx] = new_point
                    fitness[idx] = new_fit
                    if new_fit < best_val:
                        best_val = new_fit
                        best_x = new_point.copy()
                        report_best(best_val, best_x)
                # also reinitialize archive? Optionally clear
                archive = [rng.uniform(lb, ub, size=dim) for _ in range(min(archive_size, budget - evals))]
                archive = [a for a in archive if evals < budget]  # dummy, not evaluated
                stagnation_counter = 0
                gen = 0
            gen += 1

        return best_val, best_x