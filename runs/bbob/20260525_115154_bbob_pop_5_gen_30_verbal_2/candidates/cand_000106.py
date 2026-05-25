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
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        bounds_range = ub - lb

        # Initial population size
        pop_size_start = max(4 * dim, 10)
        pop_size_end = max(3, dim)
        pop_size = pop_size_start

        # Initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
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
        diversity_threshold = 0.01 * np.mean(bounds_range)

        def local_refine(x0, budget_local):
            nonlocal evals, best_val, best_x
            if budget_local <= 0 or dim == 1:
                return
            # Nelder-Mead simplex (simple version)
            n = dim
            simplex = np.zeros((n+1, n))
            simplex[0] = x0.copy()
            for i in range(n):
                step = 0.05 * bounds_range[i]
                point = x0.copy()
                point[i] += step
                point = np.clip(point, lb, ub)
                simplex[i+1] = point
            fsim = np.full(n+1, np.inf)
            for i in range(n+1):
                if evals >= budget:
                    return
                fsim[i] = func(simplex[i])
                evals += 1
                if fsim[i] < best_val:
                    best_val = fsim[i]
                    best_x = simplex[i].copy()
                    report_best(best_val, best_x)
            # Sort simplex
            order = np.argsort(fsim)
            simplex = simplex[order]
            fsim = fsim[order]
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            used_local = n+1
            while used_local < budget_local:
                # centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                # reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget:
                    return
                fr = func(xr)
                evals += 1
                used_local += 1
                if fr < fsim[0]:
                    # expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= budget:
                        return
                    fe = func(xe)
                    evals += 1
                    used_local += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fsim[-1] = fe
                    else:
                        simplex[-1] = xr
                        fsim[-1] = fr
                elif fr < fsim[-2]:
                    simplex[-1] = xr
                    fsim[-1] = fr
                else:
                    # contraction
                    if fr < fsim[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    if evals >= budget:
                        return
                    fc = func(xc)
                    evals += 1
                    used_local += 1
                    if fc < fsim[-1]:
                        simplex[-1] = xc
                        fsim[-1] = fc
                    else:
                        # shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            if evals >= budget:
                                return
                            fsim[i] = func(simplex[i])
                            evals += 1
                            used_local += 1
                            if fsim[i] < best_val:
                                best_val = fsim[i]
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)
                order = np.argsort(fsim)
                simplex = simplex[order]
                fsim = fsim[order]
            # Update best from simplex
            if fsim[0] < best_val:
                best_val = fsim[0]
                best_x = simplex[0].copy()
                report_best(best_val, best_x)

        while evals < budget:
            progress = evals / budget
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            pbest_ratio = 0.2 - 0.15 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            scale_F = 0.2 - 0.15 * progress
            scale_CR = 0.2 - 0.15 * progress
            archive_size = pop_size

            sort_idx = np.argsort(fitness)[:pop_size]
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
                CR_i = np.clip(CR_i, 0, 1)

                # Select pbest (different from i)
                cand = [idx for idx in pbest_pool if idx != i]
                if not cand:
                    cand = pbest_pool
                pbest_idx = rng.choice(cand)

                # Select r1 from population (exclude i and pbest)
                candidates_r1 = [j for j in range(pop_size) if j not in (i, pbest_idx)]
                if len(candidates_r1) == 0:
                    continue
                r1 = rng.choice(candidates_r1)

                # Select r2 from population and archive
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

                # Mutation
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
            if len(successful_F) > 0:
                sum_F = np.sum(successful_F)
                sum_F2 = np.sum(np.square(successful_F))
                if sum_F > 0:
                    mu_F = sum_F2 / sum_F
                mu_CR = np.mean(successful_CR)

            # Stagnation and diversity detection
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            remaining_evals = budget - evals
            # Compute population diversity
            if pop_size > 1:
                pop_center = np.mean(pop, axis=0)
                distances = np.mean(np.sqrt(np.sum((pop - pop_center)**2, axis=1)))
            else:
                distances = 0.0

            # Dynamic diversity threshold
            dyn_thresh = diversity_threshold * (1 - 0.5 * progress)
            restart = False
            if remaining_evals > 0:
                stagnation_threshold = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= stagnation_threshold:
                    restart = True
                if distances < dyn_thresh:
                    restart = True

            if restart and evals < budget:
                # Pop perturbation std
                pop_std = np.std(pop, axis=0)
                pop_std = np.maximum(pop_std, 1e-12)

                # Restart: keep best, reinitialize others
                new_pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
                new_pop[0] = best_x
                for i in range(1, pop_size):
                    if rng.rand() < 0.5:
                        noise = rng.normal(0, pop_std, size=dim)
                        new_pop[i] = np.clip(new_pop[i] + noise, lb, ub)
                    else:
                        radius = 0.1 * bounds_range
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

                # After restart, perform local refinement with 5% of remaining budget
                if remaining_evals > 0:
                    local_budget = max(1, int(0.05 * remaining_evals))
                    local_refine(best_x, local_budget)

                # Reset parameters
                mu_F = 0.5
                mu_CR = 0.5
                archive = []
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x