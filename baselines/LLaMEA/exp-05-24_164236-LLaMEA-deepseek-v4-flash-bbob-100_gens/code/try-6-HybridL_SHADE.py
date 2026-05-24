import numpy as np

class HybridL_SHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim

        # ----- Initialization (scrambled Latin Hypercube) -----
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Scrambled LHS: for each dimension, randomly permute the ranks
        lhs = np.random.rand(pop_size, dim)
        for j in range(dim):
            lhs[:, j] = (np.random.permutation(pop_size) + 0.5) / pop_size
        pop = lb + lhs * (ub - lb)

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        archive = []                     # stores replaced parents
        archive_size = pop_size
        success_F = []
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        max_gen = int(self.budget / pop_size * 2)

        # ----- Auxiliary functions for weighted Lehmer mean -----
        def weighted_lehmer(values, weights):
            w = np.array(weights)
            v = np.array(values)
            if len(v) == 0:
                return 0.5
            return np.sum(w * v**2) / max(1e-12, np.sum(w * v))

        def weighted_mean(values, weights):
            w = np.array(weights)
            v = np.array(values)
            if len(v) == 0:
                return 0.8
            return np.sum(w * v) / max(1e-12, np.sum(w))

        # ----- Main loop -----
        while evals < self.budget:
            gen += 1
            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                # Shrink archive to archive_size (if larger)
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            success_F_gen = []
            success_CR_gen = []
            delta_f_gen = []  # fitness improvements for weighting

            # Select current memory entries
            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            # ----- Mutation / Crossover -----
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # pbest selection (decreasing rate)
                p = max(0.2, 0.2 * (1 - gen / max_gen))
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # Choose two distinct individuals from pop + archive
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    x1 = archive[r1 - pop_size] if r1 >= pop_size else pop[r1]
                    x2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]
                else:
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    x1, x2 = pop[r1], pop[r2]

                # Adaptive parameters
                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                # Mutation (current-to-pbest/1)
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x1 - x2)
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Store improvement weight (absolute difference)
                    delta = fitness[i] - f_trial
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    delta_f_gen.append(max(delta, 1e-12))  # avoid zero weights
                    # Archive management
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        arch_idx = np.random.randint(len(archive))
                        archive[arch_idx] = pop[i].copy()
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # ----- Update memory with weighted Lehmer means -----
            if len(success_F_gen) > 0:
                weights = np.array(delta_f_gen)
                F_lehmer = weighted_lehmer(success_F_gen, weights)
                CR_mean = weighted_mean(success_CR_gen, weights)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ----- Adaptive periodic local search (Nelder-Mead) -----
            do_local = (gen % 5 == 0) or (stagnation_counter >= 3)
            if evals < self.budget and do_local:
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # Build initial simplex around best
                step = (ub - lb) * 0.05
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = x_best
                for k in range(dim):
                    x = x_best.copy()
                    x[k] = np.clip(x[k] + step[k], lb[k], ub[k])
                    simplex[k+1] = x
                # Evaluate simplex (except best already known)
                f_simplex = np.array([f_best] + [func(simplex[i]) for i in range(1, dim+1)])
                evals += dim
                # Nelder-Mead iterations (limited budget)
                nm_evals = 0
                max_nm_evals = min(30 * dim, self.budget - evals)
                restarts_nm = 0
                while nm_evals < max_nm_evals and evals < self.budget:
                    # Order
                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]
                    centroid = np.mean(simplex[:-1], axis=0)
                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = func(xr)
                    evals += 1
                    nm_evals += 1
                    if fr < f_simplex[0]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe)
                        evals += 1
                        nm_evals += 1
                        if fe < fr:
                            simplex[-1] = xe
                            f_simplex[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_simplex[-1] = fr
                    elif fr < f_simplex[-2]:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                    else:
                        # Contraction
                        if fr < f_simplex[-1]:
                            xc = centroid + 0.5 * (centroid - simplex[-1])
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1
                            nm_evals += 1
                            if fc < fr:
                                simplex[-1] = xc
                                f_simplex[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim+1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    f_simplex[i] = func(simplex[i])
                                    evals += 1
                                    nm_evals += 1
                        else:
                            xc = centroid - 0.5 * (centroid - simplex[-1])
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1
                            nm_evals += 1
                            if fc < f_simplex[-1]:
                                simplex[-1] = xc
                                f_simplex[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim+1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    f_simplex[i] = func(simplex[i])
                                    evals += 1
                                    nm_evals += 1
                    # Update global best
                    min_idx = np.argmin(f_simplex)
                    if f_simplex[min_idx] < self.f_opt:
                        self.f_opt = f_simplex[min_idx]
                        self.x_opt = simplex[min_idx].copy()
                    # If simplex is too flat, restart with smaller step
                    if f_simplex[-1] - f_simplex[0] < 1e-12:
                        restarts_nm += 1
                        if restarts_nm > 3:
                            break
                        # Rebuild simplex around best with smaller step
                        step = (ub - lb) * 0.01
                        simplex[0] = self.x_opt.copy()
                        for k in range(dim):
                            x = simplex[0].copy()
                            x[k] = np.clip(x[k] + step[k], lb[k], ub[k])
                            simplex[k+1] = x
                        f_simplex = np.array([self.f_opt] + [func(simplex[i]) for i in range(1, dim+1)])
                        evals += dim
                        nm_evals += dim
                # Replace worst in population with best from NM
                worst_idx = np.argmax(fitness)
                if self.f_opt < fitness[worst_idx]:
                    pop[worst_idx] = self.x_opt.copy()
                    fitness[worst_idx] = self.f_opt

            # ----- Stagnation detection -----
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # ----- Cauchy-based restart on severe stagnation -----
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                # Keep best solution
                best_f = self.f_opt
                best_x = self.x_opt.copy()
                # Reinitialize part of population with Cauchy noise
                n_restart = max(2, int(0.7 * pop_size))
                idx_restart = np.random.choice(pop_size, n_restart, replace=False)
                for idx in idx_restart:
                    # Cauchy scale decreases with generation
                    scale = (ub - lb) * max(0.1, 0.5 * (1 - gen / max_gen))
                    # Generate Cauchy perturbation
                    noise = np.random.standard_cauchy(dim)
                    noise = np.clip(noise, -10, 10)  # avoid extreme values
                    pop[idx] = best_x + scale * noise
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reinitialize remaining population uniformly
                n_uniform = pop_size - n_restart
                if n_uniform > 0:
                    uni_idx = np.setdiff1d(np.arange(pop_size), idx_restart)
                    uni_pop = lb + np.random.rand(n_uniform, dim) * (ub - lb)
                    for i, idx in enumerate(uni_idx):
                        pop[idx] = uni_pop[i]
                        if evals < self.budget:
                            fitness[idx] = func(pop[idx])
                            evals += 1
                            if fitness[idx] < self.f_opt:
                                self.f_opt = fitness[idx]
                                self.x_opt = pop[idx].copy()
                # Reset stagnation and memory
                stagnation_counter = 0
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive = []

        return self.f_opt, self.x_opt