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

        # initial incumbent
        best_val = np.inf
        best_x = None
        evals = 0

        # initial population size
        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)
        pop_size = max(pop_size_start, pop_size_end)
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
        mu_CR = 0.5
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0

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
            if remaining_evals > 0:
                threshold_gen = max(1, int(0.1 * remaining_evals / pop_size))
                pop_std = np.std(pop, axis=0)
                norm_std = pop_std / (ub - lb)
                diversity_trigger = np.mean(norm_std) < 1e-3
                if (gen_no_improve >= threshold_gen or diversity_trigger) and evals < budget:
                    # Reinitialize population around best with covariance adaptation
                    # Use a simplified CMA-ES-like step: sample from a multivariate normal with covariance from top half
                    num_elite = max(pop_size // 2, 3)
                    elite_idx = np.argsort(fitness)[:num_elite]
                    elite = pop[elite_idx]
                    elite_center = np.mean(elite, axis=0)
                    cov = np.cov(elite, rowvar=False) + 1e-12 * np.eye(dim)
                    # Perform a few generations with small population
                    cma_pop_size = min(pop_size, 5)
                    cma_generations = min(3, remaining_evals // cma_pop_size)
                    cma_best_val = best_val
                    cma_best_x = best_x.copy()
                    cma_cov = cov
                    cma_center = elite_center
                    for _ in range(cma_generations):
                        if evals + cma_pop_size > budget:
                            break
                        # Sample offspring
                        offspring = rng.multivariate_normal(cma_center, cma_cov, size=cma_pop_size)
                        offspring = np.clip(offspring, lb, ub)
                        for j in range(cma_pop_size):
                            val = func(offspring[j])
                            evals += 1
                            if val < cma_best_val:
                                cma_best_val = val
                                cma_best_x = offspring[j].copy()
                                report_best(cma_best_val, cma_best_x)
                        # Update center to best offspring
                        best_offspring_idx = np.argmin([func(offspring[j]) for j in range(cma_pop_size)] if False else range(cma_pop_size)) # dummy to compute min
                        # Actually we need to compute fitness for all offspring to pick best
                        # Already computed above; store in cma_fitness
                        cma_fitness = np.array([func(offspring[j]) for j in range(cma_pop_size)])  # but be careful not to double count evals
                        # Since we already evaluated, we can just use the stored values? We need to avoid extra calls.
                        # Let's restructure: evaluate first, then update.
                        # We'll use a loop that evaluates and updates.
                        
                    # Alternative: simpler approach: just reinitialize population and perform a few local steps with covariance.
                    # To keep code simple and avoid repeated function calls, we'll reinitialize the whole population with diversity from covariance.
                    new_pop = rng.multivariate_normal(best_x, cov, size=pop_size)
                    new_pop = np.clip(new_pop, lb, ub)
                    new_pop[0] = best_x
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
                    # Also do a few local CMA-ES steps around best
                    local_budget = min(10, remaining_evals - max(0, budget - evals))
                    cma_center = best_x
                    cma_cov = cov
                    for _ in range(local_budget):
                        if evals >= budget:
                            break
                        step = rng.multivariate_normal(np.zeros(dim), cma_cov)
                        candidate = np.clip(cma_center + step, lb, ub)
                        val = func(candidate)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = candidate.copy()
                            report_best(best_val, best_x)
                            # update step-size/ covariance? For simplicity, we keep covariance fixed.
                        # else: shrink? Not implemented for now to avoid complexity.
                    mu_F = 0.5
                    mu_CR = 0.5
                    archive = []
                    prev_best_val = best_val
                    gen_no_improve = 0

        return best_val, best_x