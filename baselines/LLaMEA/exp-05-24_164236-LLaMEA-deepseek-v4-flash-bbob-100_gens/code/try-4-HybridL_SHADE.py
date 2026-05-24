import numpy as np

class HybridL_SHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim

        # Latin Hypercube initialization
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
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

        archive = []  # store replaced parents for diversity
        archive_size = pop_size
        success_F = []
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        max_gen = int(self.budget / pop_size * 2)

        while evals < self.budget:
            gen += 1
            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]]
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                # shrink archive
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            success_F_gen = []
            success_CR_gen = []

            r = np.random.randint(mem_size)
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
                    # if r1 or r2 refer to archive, use archive index
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

                # Adaptive parameters
                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
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
                    # Store parent in archive (if space)
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # replace random archive element
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
                mem_idx = (mem_idx + 1) % mem_size

            # --- Nelder-Mead local search on best (periodic) ---
            if evals < self.budget and (gen % 5 == 0 or stagnation_counter >= 3):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # Create initial simplex around best
                step = (ub - lb) * 0.05
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = x_best
                for k in range(dim):
                    x = x_best.copy()
                    x[k] = np.clip(x[k] + step[k], lb[k], ub[k])
                    simplex[k+1] = x
                # Evaluate simplex
                f_simplex = np.array([f_best] + [func(simplex[i]) for i in range(1, dim+1)])
                evals += dim
                # Nelder-Mead iterations (limited)
                nm_evals = 0
                max_nm_evals = min(50 * dim, self.budget - evals)
                while nm_evals < max_nm_evals:
                    # Order simplex
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
                        xe = centroid + 2 * (centroid - simplex[-1])
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
                # Replace worst in population with best from NM
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

        return self.f_opt, self.x_opt