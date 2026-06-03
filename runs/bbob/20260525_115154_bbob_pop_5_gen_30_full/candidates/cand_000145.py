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
        pop_size_start = max(4 * dim, 10)
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
            # Update population size (linear decrease)
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

            # Population diversity
            if pop_size > 1:
                pop_center = np.mean(pop, axis=0)
                distances = np.mean(np.sqrt(np.sum((pop - pop_center)**2, axis=1)))
            else:
                distances = 0.0
            diversity_low = distances < diversity_threshold_base * (1 + progress)

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

                # Mutation scheme: current-to-pbest/1 normally, but if diversity low and budget allows, use rand/1 with probability 0.5
                use_rand1 = diversity_low and rng.rand() < 0.5

                if use_rand1:
                    # rand/1: select three distinct random indices from population
                    indices = rng.choice(pop_size, size=3, replace=False)
                    r0, r1, r2 = indices[0], indices[1], indices[2]
                    mutant = pop[r0] + F_i * (pop[r1] - pop[r2])
                else:
                    # current-to-pbest/1
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

            remaining_evals = budget - evals
            restart = False
            if remaining_evals > 0:
                stagnation_limit = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= stagnation_limit:
                    restart = True
                if distances < diversity_threshold_base * (1 + progress):
                    restart = True

            if restart and evals < budget:
                # Local refinement with short CMA-ES (BIPOP-like)
                # Use small population: 4 + floor(3*log(dim))
                cma_pop_size = max(4, int(4 + 3 * np.log(dim)))
                # Budget for CMA-ES: 10% of remaining but at least 20 evals
                cma_budget = max(20, int(0.1 * remaining_evals))
                cma_budget = min(cma_budget, remaining_evals)
                if cma_budget > 0 and cma_pop_size > 0:
                    # Initialize CMA-ES with current best point
                    xmean = best_x.copy()
                    sigma = 0.2 * (ub - lb)  # step size
                    # Diagonal covariance (simplified)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    C = np.eye(dim)
                    invsqrtC = np.eye(dim)
                    eigeneval = 0
                    chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                    counteval = 0
                    while counteval < cma_budget:
                        # Generate offspring
                        arz = rng.randn(cma_pop_size, dim)
                        arx = xmean + sigma * (arz @ np.diag(D) @ B.T)
                        arx = np.clip(arx, lb, ub)
                        arfitness = np.full(cma_pop_size, np.inf)
                        for k in range(cma_pop_size):
                            if evals >= budget:
                                break
                            arfitness[k] = func(arx[k])
                            evals += 1
                            counteval += 1
                            if arfitness[k] < best_val:
                                best_val = arfitness[k]
                                best_x = arx[k].copy()
                                report_best(best_val, best_x)
                        if evals >= budget:
                            break
                        # Sort and update mean
                        sort_idx = np.argsort(arfitness)
                        xold = xmean.copy()
                        xmean = np.mean(arx[sort_idx[:cma_pop_size//2]], axis=0)
                        # Update evolution paths (simplified, no cumulation)
                        # Not fully correct but sufficient for short run
                        pc = (1 - 1/cma_pop_size) * pc + np.sqrt(1 - 1/cma_pop_size) * (xmean - xold) / sigma
                        ps = (1 - 1/cma_pop_size) * ps + np.sqrt(1 - 1/cma_pop_size) * invsqrtC @ (xmean - xold) / sigma
                        # Update covariance matrix (simplified, diagonal)
                        # For simplicity, just adjust step size based on ps
                        sigma = sigma * np.exp(1/cma_pop_size * (np.linalg.norm(ps) - chiN) / chiN)
                        sigma = max(sigma, 1e-12)
                        sigma = min(sigma, 0.5 * np.mean(ub - lb))
                    # CMA-ES local refinement done

                # Reinitialize population (JADE restart)
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