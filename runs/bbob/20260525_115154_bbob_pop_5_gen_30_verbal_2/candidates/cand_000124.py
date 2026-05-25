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
        lb = func.bounds.lb.astype(np.float64)
        ub = func.bounds.ub.astype(np.float64)

        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)
        pop_size = pop_size_start

        pop = rng.uniform(lb, ub, size=(pop_size, dim))
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
        diversity_threshold_base = 0.01 * np.mean(ub - lb)

        while evals < budget:
            progress = evals / budget
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            pbest_ratio = 0.2 - 0.15 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            scale_F = 0.2 - 0.15 * progress
            scale_CR = 0.2 - 0.15 * progress
            archive_size = pop_size

            sort_idx = np.argsort(fitness)
            pbest_pool = sort_idx[:num_pbest]

            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                F_i = mu_F + scale_F * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + scale_F * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                CR_i = mu_CR + scale_CR * rng.randn()
                CR_i = np.clip(CR_i, 0.0, 1.0)

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
                if isinstance(candidates_r2[pick], np.ndarray):
                    r2 = candidates_r2[pick]
                else:
                    r2 = pop[candidates_r2[pick]]

                # Optionally use rand/1 for diversity
                if pop_size > 1:
                    pop_center = np.mean(pop, axis=0)
                    distances = np.mean(np.sqrt(np.sum((pop - pop_center)**2, axis=1)))
                else:
                    distances = 0.0
                diversity_threshold = diversity_threshold_base * (1 + progress)
                use_rand1 = distances < diversity_threshold and rng.rand() < 0.3

                if use_rand1:
                    # rand/1
                    candidates_rand = [j for j in range(pop_size) if j != i]
                    r1_rand = rng.choice(candidates_rand)
                    candidates_r2_rand = [j for j in range(pop_size) if j not in (i, r1_rand)]
                    if archive:
                        candidates_r2_rand.extend(archive)
                    if len(candidates_r2_rand) == 0:
                        continue
                    pick2 = rng.randint(len(candidates_r2_rand))
                    if isinstance(candidates_r2_rand[pick2], np.ndarray):
                        r22 = candidates_r2_rand[pick2]
                    else:
                        r22 = pop[candidates_r2_rand[pick2]]
                    mutant = pop[r1_rand] + F_i * (pop[i] - pop[r1_rand]) + F_i * (pop[r1_rand] - r22)
                else:
                    # current-to-pbest/1
                    mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - r2)

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
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if successful_F:
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

            if pop_size > 1:
                pop_center = np.mean(pop, axis=0)
                distances = np.mean(np.sqrt(np.sum((pop - pop_center)**2, axis=1)))
            else:
                distances = 0.0

            diversity_threshold = diversity_threshold_base * (1 + progress)
            remaining_evals = budget - evals
            restart = False
            if remaining_evals > 0:
                stagnation_limit = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= stagnation_limit:
                    restart = True
                if distances < diversity_threshold:
                    restart = True

            if restart and evals < budget:
                # Local refinement with (1+1)-CMA-ES
                local_budget = max(1, int(0.05 * remaining_evals))
                if local_budget > 0:
                    sigma = 0.2 * (ub - lb)
                    mean = best_x.copy()
                    pc = np.zeros(dim)
                    sigma0 = np.mean(sigma)
                    for _ in range(local_budget):
                        if evals >= budget:
                            break
                        z = rng.randn(dim)
                        x = mean + sigma * z
                        x = np.clip(x, lb, ub)
                        val = func(x)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        if val < best_val:  # success
                            mean = x
                            pc = (1 - 1/dim) * pc + np.sqrt(1/dim*(2-1/dim)) * z
                            sigma = sigma * np.exp(0.81/dim * (np.linalg.norm(pc) - 0.3))
                        else:
                            pc = (1 - 1/dim) * pc
                            sigma = sigma * np.exp(0.81/dim * (np.linalg.norm(pc) - 0.3))
                        sigma = np.clip(sigma, 1e-12, None)

                # Reinitialize population
                pop_std = np.std(pop, axis=0) if pop_size > 1 else np.ones(dim) * 1e-12
                pop_std = np.maximum(pop_std, 1e-12)

                new_pop = rng.uniform(lb, ub, size=(pop_size, dim))
                new_pop[0] = best_x
                for i in range(1, pop_size):
                    if rng.rand() < 0.5:
                        noise = rng.normal(0, pop_std, dim)
                        new_pop[i] = np.clip(new_pop[i] + noise, lb, ub)
                    else:
                        radius = 0.1 * (ub - lb)
                        new_pop[i] = np.clip(best_x + rng.uniform(-radius, radius), lb, ub)

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

        return best_val, best_x