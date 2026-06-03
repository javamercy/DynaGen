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

        # Initial population size
        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)
        pop_size = pop_size_start

        # Initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
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

        # JADE parameters
        mu_F = 0.5
        mu_CR = 0.5
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0
        diversity_threshold_base = 0.01 * np.mean(ub - lb)

        while evals < budget:
            progress = evals / budget
            # Update population size
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            pbest_ratio = 0.2 - 0.15 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            scale_F = 0.2 - 0.15 * progress
            scale_CR = 0.2 - 0.15 * progress
            archive_size = pop_size

            # Sort for pbest
            sort_idx = np.argsort(fitness)
            pbest_pool = sort_idx[:num_pbest]

            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # Generate F_i
                F_i = mu_F + scale_F * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + scale_F * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                # Generate CR_i
                CR_i = mu_CR + scale_CR * rng.randn()
                CR_i = np.clip(CR_i, 0.0, 1.0)

                # Select mutant components
                # Determine whether to use rand/1 (for diversity)
                use_rand1 = rng.rand() < 0.2 * (1 - progress)  # higher early
                if use_rand1:
                    # rand/1
                    candidates = [j for j in range(pop_size) if j != i]
                    if len(candidates) < 3:
                        continue
                    r0, r1, r2 = rng.choice(candidates, size=3, replace=False)
                    mutant = pop[r0] + F_i * (pop[r1] - pop[r2])
                else:
                    # current-to-pbest/1 with archive
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

                    mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - r2)

                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluation
                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    # Add parent to archive
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

            # Update parameter means
            if successful_F:
                sum_F = np.sum(successful_F)
                sum_F2 = np.sum(np.square(successful_F))
                if sum_F > 0:
                    mu_F = sum_F2 / sum_F
                mu_CR = np.mean(successful_CR)

            # Stagnation detection
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            # Population diversity
            if pop_size > 1:
                pop_center = np.mean(pop, axis=0)
                distances = np.mean(np.sqrt(np.sum((pop - pop_center)**2, axis=1)))
            else:
                distances = 0.0

            # Dynamic diversity threshold
            diversity_threshold = diversity_threshold_base * (1 + progress)

            # Restart conditions
            remaining_evals = budget - evals
            restart = False
            if remaining_evals > 0:
                stagnation_limit = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= stagnation_limit:
                    restart = True
                if distances < diversity_threshold:
                    restart = True

            if restart and evals < budget:
                # (1+1)-CMA-ES local search on best
                local_budget = max(1, int(0.05 * remaining_evals))
                if local_budget > 0:
                    # Initialize CMA-ES parameters
                    sigma = 0.2 * np.mean(ub - lb)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    p_c = np.zeros(dim)
                    c_c = 2.0 / (dim + 2.0)
                    c_sigma = 2.0 / (dim + 2.0)
                    d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((dim-1)/dim) - 1.0)
                    chi_n = np.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim*dim))
                    p_sigma = np.zeros(dim)
                    local_best_val = best_val
                    local_best_x = best_x.copy()
                    for _ in range(local_budget):
                        if evals >= budget:
                            break
                        z = rng.randn(dim)
                        candidate = local_best_x + sigma * B @ (D * z)
                        candidate = np.clip(candidate, lb, ub)
                        val = func(candidate)
                        evals += 1
                        if val < local_best_val:
                            local_best_val = val
                            local_best_x = candidate.copy()
                            report_best(local_best_val, local_best_x)
                            # Update evolution paths
                            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * z
                            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (candidate - local_best_x) / sigma
                            # Update B and D (rank-one update)
                            delta = (candidate - local_best_x) / sigma
                            # Approximation: update diagonal D
                            D = D * np.exp(c_c / (2 * dim) * (delta**2 - 1))
                            D = np.clip(D, 0.1, 10.0)
                            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                        else:
                            p_sigma = (1 - c_sigma) * p_sigma
                        sigma = max(sigma, 1e-12)
                    if local_best_val < best_val:
                        best_val = local_best_val
                        best_x = local_best_x.copy()

                # Reinitialize population
                pop_std = np.std(pop, axis=0)
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

                # Reset JADE parameters
                mu_F = 0.5
                mu_CR = 0.5
                archive = []
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x