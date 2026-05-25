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

        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        archive = []
        archive_max = pop_size  # reduced from parent

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

        mu_F = 0.5
        mu_CR = 0.5
        p_best = 0.2
        c = 0.1
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val
        successful_F = []
        successful_CR = []
        generation = 0
        stagnation_counter = 0  # track consecutive generations without improvement

        while evals < budget:
            generation += 1
            # compute diversity measure
            if pop_size > 1:
                std_pop = np.std(pop, axis=0)
                domain_range = ub - lb
                domain_range = np.where(domain_range == 0, 1.0, domain_range)
                diversity = np.mean(std_pop / domain_range)
            else:
                diversity = 0.0

            # adapt p_best based on diversity
            if diversity < 0.1:
                p_best = min(0.5, p_best + 0.05)
            else:
                p_best = max(0.1, p_best - 0.01)

            # generate F and CR
            F = np.clip(mu_F + mu_F * rng.standard_cauchy(pop_size), 0, 1)
            F = np.where(F <= 0, 0.1, F)
            CR = np.clip(mu_CR + 0.1 * rng.standard_cauchy(pop_size), 0, 1)
            CR = np.where(CR <= 0, 0.1, CR)

            # decide whether to use archive based on stagnation
            use_archive = (stagnation_counter > 3) or (gen_no_improve > 0)

            for i in range(pop_size):
                if evals >= budget:
                    break
                pbest_size = max(2, int(pop_size * p_best))
                sorted_idx = np.argsort(fitness)
                pbest_idx = sorted_idx[:pbest_size]
                pbest = pop[pbest_idx[rng.randint(pbest_size)]]

                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b = rng.choice(candidates, size=2, replace=False)
                if use_archive and len(archive) > 0:
                    archive_idx = rng.randint(len(archive))
                    partner = archive[archive_idx]
                else:
                    partner = pop[rng.choice(candidates)]
                mutant = pop[i] + F[i] * (pbest - pop[i]) + F[i] * (pop[a] - partner)
                mutant = np.clip(mutant, lb, ub)

                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)
                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F[i])
                    successful_CR.append(CR[i])
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update success memory
            if len(successful_F) > 0:
                Lehmer = np.sum(np.array(successful_F)**2) / np.sum(np.array(successful_F))
                mu_F = (1 - c) * mu_F + c * Lehmer
                successful_F.clear()
            if len(successful_CR) > 0:
                mu_CR = (1 - c) * mu_CR + c * np.mean(successful_CR)
                successful_CR.clear()

            # track stagnation
            if best_val < prev_best_val:
                gen_no_improve = 0
                stagnation_counter = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1
                stagnation_counter += 1

            # restart if stagnation too long
            if gen_no_improve >= gen_max_restart and evals < budget:
                # reinitialize population keeping best
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                # scale based on diversity
                if pop_size > 1:
                    scale_factor = 0.3 * (1 + diversity)
                else:
                    scale_factor = 0.3
                scale = (ub - lb) * scale_factor
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
                stagnation_counter = 0
                archive.clear()

        return best_val, best_x