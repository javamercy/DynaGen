import numpy as np

class HybridL_SHADE_Refined:
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
        N_init = max(10, int(18 * np.sqrt(dim)))
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

        archive = []
        archive_size = pop_size
        success_F = []
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        max_gen = int(self.budget / pop_size * 2)

        # diversity measure threshold
        diversity_threshold = 0.05 * (ub - lb).mean()

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

            # Weighted success memory update - use weighted Lehmer mean
            success_F_gen = []
            success_CR_gen = []
            delta_f_gen = []

            # Random memory index
            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # pbest selection with adaptive rate
                p = max(0.2, 0.2 * (1 - gen / max_gen))
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # Choose two distinct random indices from pop + archive
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

                # Generate F using Cauchy distribution (as in SHADE)
                F = np.clip(np.random.cauchy(F_base, 0.1), 0.1, 1.0)
                # Generate CR using Normal distribution
                CR = np.clip(np.random.normal(CR_base, 0.1), 0.0, 1.0)

                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Store successful parameters with fitness improvement weight
                    delta_f = fitness[i] - f_trial
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    delta_f_gen.append(delta_f)
                    # Archive management: store parent
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

            # Weighted memory update using Lehmer mean (weighted by delta_f)
            if len(success_F_gen) > 0:
                w = np.array(delta_f_gen)
                # ensure weights are positive and avoid division by zero
                w = np.maximum(w, 1e-12)
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / np.sum(w * np.array(success_F_gen))
                CR_mean = np.sum(w * np.array(success_CR_gen)) / np.sum(w)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # Adaptive local search: apply when stagnation is detected
            if evals < self.budget and (stagnation_counter >= 2):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # Create small simplex around best
                step = (ub - lb) * 0.02  # smaller step
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = x_best
                for k in range(dim):
                    x = x_best.copy()
                    x[k] = np.clip(x[k] + step[k], lb[k], ub[k])
                    simplex[k+1] = x
                f_simplex = np.array([f_best] + [func(simplex[i]) for i in range(1, dim+1)])
                evals += dim
                nm_evals = 0
                max_nm_evals = min(20 * dim, self.budget - evals)  # reduce NM budget
                while nm_evals < max_nm_evals:
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
                # Replace worst in population with best from NM (if better)
                worst_idx = np.argmax(fitness)
                if self.f_opt < fitness[worst_idx]:
                    pop[worst_idx] = self.x_opt.copy()
                    fitness[worst_idx] = self.f_opt
                stagnation_counter = 0  # reset after local search

            # Stagnation detection
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Diversity-guided restart: if population too converged or stagnation severe
            if stagnation_counter > max(5, int(0.05 * max_gen)):
                # keep best solution, reinitialize rest with perturbation and some random
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                n_restart = max(int(0.8 * pop_size), 1)  # reinitialize most of population
                # reinitialize using Latin Hypercube scaled around best
                lhs = np.random.rand(n_restart, dim)
                for j in range(dim):
                    lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / n_restart
                # perturb around best with scaling
                spread = (ub - lb) * 0.5 * (1.0 - gen / max_gen)
                new_points = best_copy + spread * (2 * lhs - 1)
                new_points = np.clip(new_points, lb, ub)
                # replace random members with new points
                idx_replace = np.random.choice(pop_size, n_restart, replace=False)
                for idx, new_point in zip(idx_replace, new_points):
                    pop[idx] = new_point
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                stagnation_counter = 0
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()

        return self.f_opt, self.x_opt