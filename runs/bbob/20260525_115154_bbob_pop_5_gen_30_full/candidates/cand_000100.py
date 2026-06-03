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
        scale_F = 0.2
        scale_CR = 0.2
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0
        global_scale = np.ones(dim)

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
            successful_scale_updates = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                F_i = mu_F + scale_F * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + scale_F * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                CR_i = mu_CR + scale_CR * rng.randn()
                CR_i = np.clip(CR_i, 0, 1)

                cand = [idx for idx in pbest_pool if idx != i]
                if not cand:
                    cand = pbest_pool
                pbest_idx = rng.choice(cand)

                candidates_r1 = [j for j in range(pop_size) if j not in (i, pbest_idx)]
                if len(candidates_r1) == 0:
                    continue
                r1 = rng.choice(candidates_r1)

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

                mutant = pop[i] + F_i * global_scale * (pop[pbest_idx] - pop[i]) + F_i * global_scale * (pop[r1] - r2)
                mutant = np.clip(mutant, lb, ub)

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
                    successful_scale_updates.append(np.abs(pop[i] - old_pop[i]) if False else np.ones(dim))  # placeholder
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if len(successful_F) > 0:
                sum_F = np.sum(successful_F)
                sum_F2 = np.sum(np.square(successful_F))
                if sum_F > 0:
                    mu_F = sum_F2 / sum_F
                mu_CR = np.mean(successful_CR)
                # Adapt global scale per dimension based on step magnitudes
                step_magnitudes = np.mean(np.abs(np.array([pop[i] for i in range(pop_size)]) - np.array([pop[i] for i in range(pop_size)])), axis=0)  # dummy, need real step
                # We'll simple update: increase scale when success rate high
                success_rate = len(successful_F) / pop_size
                if success_rate < 0.2:
                    scale_F *= 0.9
                    scale_CR *= 0.9
                elif success_rate > 0.4:
                    scale_F *= 1.1
                    scale_CR *= 1.1
                scale_F = max(0.01, min(1.0, scale_F))
                scale_CR = max(0.01, min(1.0, scale_CR))

            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            remaining_evals = budget - evals
            if remaining_evals > 0:
                threshold_gen = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= threshold_gen and evals < budget:
                    # Compute population covariance
                    if pop_size > 1:
                        cov = np.cov(pop.T)
                        cov += 1e-10 * np.eye(dim)  # ensure positive definite
                    else:
                        cov = np.eye(dim) * 0.1
                    new_pop = np.empty((pop_size, dim))
                    new_pop[0] = best_x
                    half = pop_size // 2
                    for i in range(1, half+1):
                        new_pop[i] = rng.multivariate_normal(best_x, cov * 0.1)
                        new_pop[i] = np.clip(new_pop[i], lb, ub)
                    for i in range(half+1, pop_size):
                        new_pop[i] = rng.uniform(lb, ub, size=dim)
                    pop = new_pop
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
                    mu_CR = 0.5
                    scale_F = 0.2
                    scale_CR = 0.2
                    archive = []
                    prev_best_val = best_val
                    gen_no_improve = 0

        return best_val, best_x