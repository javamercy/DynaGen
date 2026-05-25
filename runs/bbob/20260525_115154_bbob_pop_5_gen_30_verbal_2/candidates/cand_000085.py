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

        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)

        pop_size = pop_size_start
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

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
        mu_CR = 0.5
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0
        success_rate = 0.5

        while evals < budget:
            progress = evals / budget
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            pbest_ratio = 0.2 - 0.15 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            archive_size = pop_size

            sort_idx = np.argsort(fitness)[:pop_size]
            pbest_pool = sort_idx[:num_pbest]

            successful_F = []
            successful_CR = []
            success_count = 0
            trial_count = 0

            for i in range(pop_size):
                if evals >= budget:
                    break

                # sample F from Cauchy with mean mu_F
                F_i = mu_F + 0.1 * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + 0.1 * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                # sample CR from normal with mean mu_CR
                CR_i = mu_CR + 0.1 * rng.randn()
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

                # select r2 from union of pop and archive
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
                trial_count += 1

                if trial_fit < fitness[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))
                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                    success_count += 1
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update mu_F and mu_CR based on success rate
            if trial_count > 0:
                current_success_rate = success_count / trial_count
                success_rate = 0.9 * success_rate + 0.1 * current_success_rate
                # adapt means: if success_rate > 0.4, increase; else decrease
                if success_rate > 0.4:
                    mu_F *= 1.05
                    mu_CR *= 1.05
                else:
                    mu_F *= 0.95
                    mu_CR *= 0.95
                mu_F = np.clip(mu_F, 0.1, 0.9)
                mu_CR = np.clip(mu_CR, 0.1, 0.9)

            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            remaining_evals = budget - evals
            if remaining_evals > 0:
                threshold_gen = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= threshold_gen and evals < budget:
                    # covariance-based restart
                    pop_tmp = pop[:pop_size]
                    fit_tmp = fitness[:pop_size]
                    # use top half of population
                    top_idx = np.argsort(fit_tmp)[:max(2, pop_size//2)]
                    top_pop = pop_tmp[top_idx]
                    if len(top_pop) > 1:
                        cov = np.cov(top_pop, rowvar=False) + 1e-9 * np.eye(dim)
                        mean = best_x
                    else:
                        cov = np.eye(dim) * 0.01
                        mean = best_x
                    # generate new population
                    new_pop = np.zeros_like(pop_tmp)
                    new_pop[0] = best_x
                    for i in range(1, pop_size):
                        sample = rng.multivariate_normal(mean, cov)
                        sample = np.clip(sample, lb, ub)
                        new_pop[i] = sample
                    pop = new_pop
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
                    mu_CR = 0.5
                    archive = []
                    prev_best_val = best_val
                    gen_no_improve = 0
                    success_rate = 0.5

        return best_val, best_x