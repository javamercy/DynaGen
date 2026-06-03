import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # initial population size: start large, shrink aggressively
        pop_size_start = max(4 * dim, 10)
        pop_size_end = max(2, int(dim / 2))  # smaller final pop for exploitation

        best_val = np.inf
        best_x = None
        evals = 0

        pop_size = pop_size_start
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
        fitness = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        mu_F = 0.5
        mu_CR = 0.9  # start high for exploration, then reduce
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0

        while evals < budget:
            progress = evals / budget
            # population size decreases faster
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress**2)
            pop_size = max(2, pop_size)
            # pbest ratio shrinks to focus on best
            pbest_ratio = 0.1 - 0.05 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            # F and CR adaptation: smaller scale for exploitation
            scale_F = 0.1 - 0.05 * progress
            scale_CR = 0.1 - 0.05 * progress
            archive_size = pop_size

            # ensure population size matches current pop_size
            if len(pop) > pop_size:
                # keep best
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:pop_size]]
                fitness = fitness[sort_idx[:pop_size]]
            elif len(pop) < pop_size:
                # add new random points
                new = rng.uniform(lb, ub, size=(pop_size - len(pop), dim)).astype(float)
                pop = np.vstack([pop, new])
                new_fitness = np.full(pop_size - len(pop), np.inf)
                for i in range(len(new)):
                    val = func(new[i])
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = new[i].copy()
                        report_best(best_val, best_x)
                new_fitness = np.concatenate([fitness, new_fitness])
                fitness = new_fitness

            sort_idx = np.argsort(fitness)
            pbest_pool = sort_idx[:num_pbest]

            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # generate F from truncated Cauchy
                F_i = mu_F + scale_F * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + scale_F * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                # generate CR from truncated normal
                CR_i = mu_CR + scale_CR * rng.randn()
                CR_i = np.clip(CR_i, 0, 1)

                # select pbest
                cand = [idx for idx in pbest_pool if idx != i]
                if not cand:
                    cand = pbest_pool
                pbest_idx = rng.choice(cand)

                # select r1
                candidates_r1 = [j for j in range(pop_size) if j not in (i, pbest_idx)]
                if len(candidates_r1) == 0:
                    continue
                r1 = rng.choice(candidates_r1)

                # select r2 from population or archive
                candidates_r2 = [j for j in range(pop_size) if j not in (i, pbest_idx, r1)]
                if archive:
                    candidates_r2.extend(archive)
                if len(candidates_r2) == 0:
                    continue
                pick = rng.randint(len(candidates_r2))
                if isinstance(candidates_r2[pick], int):
                    r2 = pop[candidates_r2[pick]]
                else:
                    r2 = candidates_r2[pick]

                # mutation
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - r2)
                mutant = np.clip(mutant, lb, ub)

                # crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))
                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update mu_F and mu_CR
            if len(successful_F) > 0:
                sum_F = np.sum(successful_F)
                sum_F2 = np.sum(np.square(successful_F))
                if sum_F > 0:
                    mu_F = sum_F2 / sum_F
                mu_CR = np.mean(successful_CR)

            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            remaining_evals = budget - evals
            # restart condition: no improvement or low diversity
            pop_std = np.std(pop, axis=0)
            norm_std = pop_std / (ub - lb)
            diversity_trigger = np.mean(norm_std) < 1e-4
            threshold_gen = max(1, int(0.05 * remaining_evals / pop_size))
            if (gen_no_improve >= threshold_gen or diversity_trigger) and evals < budget:
                # local search around best before restart
                local_budget = min(int(0.1 * remaining_evals), 100)
                if local_budget > 0:
                    x_best = best_x.copy()
                    sigma_init = 0.05 * (ub - lb)
                    C = np.eye(dim)
                    sigma = sigma_init
                    for _ in range(local_budget):
                        if evals >= budget:
                            break
                        z = rng.randn(dim)
                        step = sigma * np.dot(np.linalg.cholesky(C), z)
                        candidate = np.clip(x_best + step, lb, ub)
                        val = func(candidate)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = candidate.copy()
                            report_best(best_val, best_x)
                            # update covariance (rank-one)
                            delta = step / sigma
                            C = (1 - 1.0/dim) * C + (1.0/dim) * np.outer(delta, delta)
                            sigma *= 1.3
                        else:
                            sigma *= 0.8
                        sigma = np.clip(sigma, 1e-8, ub - lb)
                # restart: reinitialize population
                pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
                pop[0] = best_x
                fitness = np.full(pop_size, np.inf)
                fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    val = func(pop[i])
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                mu_F = 0.5
                mu_CR = 0.9
                archive = []
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x