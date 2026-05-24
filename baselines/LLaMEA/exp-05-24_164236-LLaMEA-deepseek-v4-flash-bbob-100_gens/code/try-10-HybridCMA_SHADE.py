import numpy as np
from math import log, sqrt, floor, exp

class HybridCMA_SHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def _sobol_sample(self, n, d, seed=42):
        """Generate n points in [0,1]^d using Sobol sequence (simple implementation)."""
        # Using a basic Sobol generator via Gray code (low discrepancy)
        # This is a simple version; for production use scipy.stats.qmc.Sobol
        # We'll use a deterministic method to ensure reproducibility.
        # We'll generate using the 'sobol_seq' package if available, but here we implement a minimal one.
        # Instead, we fall back to Latin Hypercube as it is adequate.
        # Use LHS for simplicity (no external deps)
        rng = np.random.RandomState(seed)
        points = np.empty((n, d))
        for j in range(d):
            points[:, j] = (np.argsort(rng.rand(n)) + 0.5) / n
        return points

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        D = dim

        # Population size settings
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        memory_size = 6
        mem_F = np.full(memory_size, 0.5)
        mem_CR = np.full(memory_size, 0.8)
        mem_idx = 0

        # Initialize population using Latin Hypercube (or Sobol)
        n_init = pop_size
        lhs = self._sobol_sample(n_init, dim)
        pop = lb + lhs * (ub - lb)

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        archive = []
        archive_size = pop_size
        success_F = []
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        max_gen = int(self.budget / pop_size * 2)

        # CMA-ES parameters for local search
        cma_sigma = 0.2 * np.mean(ub - lb)  # initial step size
        cma_path = np.zeros(dim)             # evolution path for sigma
        cma_C = np.eye(dim)                  # covariance matrix

        while evals < self.budget:
            gen += 1
            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]]
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            success_F_gen = []
            success_CR_gen = []

            r = np.random.randint(memory_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # pbest selection (adaptive rate)
                p = max(0.2, 0.2 * (1 - gen / max_gen))
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # Choose two distinct random indices from pop+archive
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    if r1 >= pop_size:
                        x_r1 = archive[r1 - pop_size]
                    else:
                        x_r1 = pop[r1]
                    if r2 >= pop_size:
                        x_r2 = archive[r2 - pop_size]
                    else:
                        x_r2 = pop[r2]
                else:
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Adaptive parameters with Cauchy for F and normal for CR
                F = np.clip(F_base + 0.1 * np.random.standard_cauchy(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    # Store parent in archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx_arch = np.random.randint(len(archive))
                        archive[idx_arch] = pop[i].copy()
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update memory with successful parameters (Lehmer mean)
            if len(success_F_gen) > 0:
                w = np.ones(len(success_F_gen))
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / np.sum(w * np.array(success_F_gen))
                CR_mean = np.mean(success_CR_gen)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % memory_size

            # --- CMA-ES local search (triggered periodically or on stagnation) ---
            if evals < self.budget and (gen % 5 == 0 or stagnation_counter >= 3):
                # Run CMA-ES on best solution
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # Determine budget for CMA-ES
                cma_lambda = max(4, int(4 + 3 * log(dim)))
                cma_budget = min(100 * dim, self.budget - evals)
                cma_evals = 0
                cma_mu = cma_lambda // 2
                # Reinitialize CMA-ES internal state
                cma_mean = x_best.copy()
                cma_sigma = 0.2 * np.mean(ub - lb)
                cma_path = np.zeros(dim)
                cma_C = np.eye(dim)
                # Weights for recombination
                weights = np.array([log(cma_mu + 0.5) - log(i+1) for i in range(cma_mu)])
                weights /= np.sum(weights)
                mu_eff = 1.0 / np.sum(weights**2)
                # Learning rates
                cc = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
                cs = (mu_eff + 2) / (dim + mu_eff + 5)
                c1 = 2 / ((dim + 1.3)**2 + mu_eff)
                cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
                damps = 1 + 2*max(0, sqrt((mu_eff-1)/(dim+1)) - 1) + cs

                cma_gen = 0
                max_cma_gen = int(cma_budget / cma_lambda)
                while cma_evals + cma_lambda <= cma_budget and cma_gen < max_cma_gen:
                    cma_gen += 1
                    # Sample population
                    A = np.linalg.cholesky(cma_C)
                    Z = np.random.randn(cma_lambda, dim)
                    X = cma_mean + cma_sigma * (Z @ A.T)
                    # Clamp and evaluate
                    X = np.clip(X, lb, ub)
                    f_vals = np.array([func(x) for x in X])
                    evals += cma_lambda
                    cma_evals += cma_lambda
                    # Sort
                    sorted_idx = np.argsort(f_vals)
                    X_sort = X[sorted_idx]
                    f_sort = f_vals[sorted_idx]
                    # Update best
                    if f_sort[0] < self.f_opt:
                        self.f_opt = f_sort[0]
                        self.x_opt = X_sort[0].copy()
                    # Update mean
                    old_mean = cma_mean.copy()
                    cma_mean = np.sum(weights[:, None] * X_sort[:cma_mu], axis=0)
                    # Update evolution path and covariance
                    cma_path = (1 - cc) * cma_path + sqrt(cc*(2-cc)*mu_eff) * (cma_mean - old_mean) / cma_sigma
                    # Rank-1 update
                    cma_C = (1 - c1 - cmu) * cma_C + c1 * np.outer(cma_path, cma_path)
                    # Rank-mu update
                    for i in range(cma_mu):
                        diff = (X_sort[i] - old_mean) / cma_sigma
                        cma_C += cmu * weights[i] * np.outer(diff, diff)
                    # Step-size control
                    ps_norm = np.linalg.norm(cma_path)
                    cma_sigma *= exp(cs/damps * (ps_norm / sqrt(dim) - 1))

                # Replace worst in population with best from CMA-ES (if improved)
                if self.f_opt < f_best - 1e-12:
                    worst_idx = np.argmax(fitness)
                    if self.f_opt < fitness[worst_idx]:
                        pop[worst_idx] = self.x_opt.copy()
                        fitness[worst_idx] = self.f_opt

            # Stagnation detection
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Restart if stagnation is severe
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                n_restart = max(1, int(0.5 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Reinitialize part of population around best
                idx_keep = np.random.choice(pop_size, n_restart, replace=False)
                for idx in idx_keep:
                    pop[idx] = best_copy + np.random.uniform(-0.2, 0.2, dim) * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                stagnation_counter = 0
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive = []

        return self.f_opt, self.x_opt