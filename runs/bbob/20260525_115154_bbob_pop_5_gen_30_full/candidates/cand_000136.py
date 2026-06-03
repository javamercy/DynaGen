import numpy as np
from numpy.linalg import norm

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)
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
                # Short separable CMA-ES local search
                cma_budget = int(0.1 * remaining_evals)
                if cma_budget > 0:
                    # CMA-ES parameters
                    lambda_cma = max(4, 4 + int(3 * np.log(dim)))
                    lambda_cma = min(lambda_cma, cma_budget)  # at most budget
                    mu_cma = lambda_cma // 2
                    sigma = 0.2 * np.mean(ub - lb)
                    weights = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
                    weights = weights / np.sum(weights)
                    mueff = 1.0 / np.sum(weights**2)
                    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
                    cs = (mueff + 2) / (dim + mueff + 5)
                    c1 = 2.0 / ((dim + 1.3)**2 + mueff)
                    cmu = min(1 - c1, 2 * (mueff - 2 + 1.0/mueff) / ((dim + 2)**2 + mueff))
                    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

                    mean = best_x.copy()
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    C = np.eye(dim)
                    invsqrtC = np.eye(dim)
                    eigeneval = 0
                    chiN = np.sqrt(dim) * (1 - 1.0/(4*dim) + 1.0/(21*dim**2))

                    while evals < budget and cma_budget > 0:
                        # Generate offspring
                        arz = rng.randn(lambda_cma, dim)
                        arx = np.zeros((lambda_cma, dim))
                        for k in range(lambda_cma):
                            arx[k] = mean + sigma * (B @ (D * arz[k]))
                        arx = np.clip(arx, lb, ub)
                        arf = np.full(lambda_cma, np.inf)
                        for k in range(lambda_cma):
                            if evals >= budget:
                                break
                            arf[k] = func(arx[k])
                            evals += 1
                            if arf[k] < best_val:
                                best_val = arf[k]
                                best_x = arx[k].copy()
                                report_best(best_val, best_x)
                            cma_budget -= 1
                        if evals >= budget:
                            break

                        # Sort by fitness
                        sort_idx = np.argsort(arf)
                        arx = arx[sort_idx]
                        arf = arf[sort_idx]

                        # Update mean
                        old_mean = mean.copy()
                        mean = np.sum(weights[:, np.newaxis] * arx[:mu_cma], axis=0)

                        # Update evolution paths
                        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean)) / sigma
                        hsig = np.sum(ps**2) / (1 - (1 - cs)**(2 * evals / lambda_cma)) / dim < 2 + 4.0/(dim+1)
                        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma

                        # Update covariance matrix
                        artmp = (arx[:mu_cma] - old_mean) / sigma
                        C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * (artmp.T @ np.diag(weights) @ artmp)

                        # Update step size
                        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

                        # Enforce symmetric and positive semidefinite
                        if evals - eigeneval > lambda_cma / (c1 + cmu) / dim / 10:
                            eigeneval = evals
                            C = np.triu(C) + np.triu(C, 1).T
                            D, B = np.linalg.eigh(C)
                            D = np.sqrt(np.abs(D))  # avoid negative eigenvalues
                            invsqrtC = B @ np.diag(1.0 / D) @ B.T

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