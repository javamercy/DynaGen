import numpy as np

class HybridL_SHADE_CMA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()  # no fixed seed for generalization
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim

        # Initial population size (Latin Hypercube)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        # Memory for F and CR
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling
        lhs = np.random.rand(pop_size, dim)
        for j in range(dim):
            lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
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

        # Multi-strategy: 0: current-to-pbest/1 (with archive), 1: current-to-rand/1, 2: rand/1
        num_strategies = 3
        strategy_prob = np.full(num_strategies, 1.0/num_strategies)
        strategy_success = np.zeros(num_strategies)
        strategy_attempts = np.ones(num_strategies)  # avoid division by zero

        while evals < self.budget:
            gen += 1
            # Linear population reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # Adaptive pbest rate (decreasing)
            p = max(0.2, 0.2 * (1 - gen / max_gen))

            # Successful parameter lists for this generation
            success_F_gen = []
            success_CR_gen = []
            # Strategy success counters
            strat_success = np.zeros(num_strategies)
            strat_attempts = np.zeros(num_strategies)

            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Choose strategy via roulette wheel
                s = np.random.choice(num_strategies, p=strategy_prob)
                strat_attempts[s] += 1

                # Generate parameters
                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                # Build mutant according to strategy
                if s == 0:  # current-to-pbest/1 with archive
                    # pbest selection
                    pbest_size = max(2, int(p * pop_size))
                    idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                    best_p = np.argmin(fitness[idx_pbest])
                    x_pbest = pop[idx_pbest[best_p]]

                    # two distinct from pop+archive
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

                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                elif s == 1:  # current-to-rand/1 (no archive)
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    mutant = pop[i] + F * (pop[r1] - pop[i]) + F * (pop[r2] - pop[r3])

                else:  # s == 2: rand/1
                    idxs = list(range(pop_size))
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    strat_success[s] += 1
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    # Store parent in archive (only for strategy 0 if using archive)
                    if s == 0:
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

            # Update strategy probabilities based on success rates
            for s in range(num_strategies):
                if strat_attempts[s] > 0:
                    success_rate = strat_success[s] / strat_attempts[s]
                    # Exponential moving average
                    strategy_success[s] = 0.9 * strategy_success[s] + 0.1 * success_rate
                    strategy_attempts[s] += 1
            # Normalize probabilities (with epsilon to avoid zero)
            total = np.sum(strategy_success) + 1e-10
            strategy_prob = (strategy_success + 1e-10) / total

            # Update memory with successful parameters (Lehmer mean)
            if len(success_F_gen) > 0:
                w = np.ones(len(success_F_gen))
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / np.sum(w * np.array(success_F_gen))
                CR_mean = np.mean(success_CR_gen)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # --- CMA-ES local search on best (periodic, on stagnation) ---
            if evals < self.budget and (gen % 5 == 0 or stagnation_counter >= 3):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # Run a short CMA-ES to improve best
                # Use a simple implementation with bounded domain
                max_nm_evals = min(100 * dim, (self.budget - evals) // 2)
                if max_nm_evals < 10:
                    continue
                # Initialize CMA-ES
                sigma = 0.2 * (ub - lb)  # per-component step size
                mean = x_best.copy()
                C = np.eye(dim)
                evals_cma = 0
                # CMA-ES parameters
                lam = int(4 + 3 * np.log(dim))  # offspring population
                mu = lam // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
                weights /= np.sum(weights)
                mueff = 1.0 / np.sum(weights**2)
                cc = (4.0 + mueff/dim) / (dim + 4.0 + 2.0*mueff/dim)
                cs = (mueff + 2.0) / (dim + mueff + 5.0)
                c1 = 2.0 / ((dim + 1.3)**2 + mueff)
                cmu = min(1 - c1, 2.0 * (mueff - 2.0 + 1.0/mueff) / ((dim + 2.0)**2 + mueff))
                damps = 1.0 + 2.0*max(0.0, np.sqrt((mueff-1)/(dim+1)) - 1.0) + cs
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                own_evals = 0
                while own_evals < max_nm_evals:
                    # Generate lambda offspring
                    A = np.linalg.cholesky(C)
                    offspring = np.array([mean + sigma * A @ np.random.randn(dim) for _ in range(lam)])
                    offspring = np.clip(offspring, lb, ub)
                    fits = np.array([func(x) for x in offspring])
                    own_evals += lam
                    evals += lam
                    # Select mu best
                    idx_sorted = np.argsort(fits)
                    x_sel = offspring[idx_sorted[:mu]]
                    # Update mean
                    old_mean = mean.copy()
                    mean = np.sum(weights[:, None] * x_sel, axis=0)
                    # Update evolution paths
                    z_mean = np.linalg.solve(A, (mean - old_mean) / sigma)
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * z_mean
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (A @ z_mean)
                    # Update covariance matrix
                    C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                    dC = np.zeros((dim, dim))
                    for i in range(mu):
                        diff = (x_sel[i] - old_mean) / sigma
                        dC += cmu * weights[i] * np.outer(diff, diff)
                    C += dC
                    # Step size control
                    sigma *= np.exp(cs / damps * (np.linalg.norm(ps) - dim**0.5) / (dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))))
                    # Ensure C positive definite
                    C = (C + C.T) / 2
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-12:
                        C += 1e-12 * np.eye(dim)
                    # Update best solution
                    best_idx = np.argmin(fits)
                    if fits[best_idx] < self.f_opt:
                        self.f_opt = fits[best_idx]
                        self.x_opt = offspring[best_idx].copy()
                # Replace worst individual in population with best from CMA-ES (if better)
                if self.f_opt < f_best:
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
                # keep best solution
                n_restart = max(1, int(0.5 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Reinitialize population around best with random perturbation
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
                # Reset memory
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                # Clear archive
                archive = []
                # Reset strategy probabilities
                strategy_prob = np.full(num_strategies, 1.0/num_strategies)
                strategy_success = np.zeros(num_strategies)
                strategy_attempts = np.ones(num_strategies)

        return self.f_opt, self.x_opt