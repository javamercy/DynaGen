import numpy as np

class Refined_HybridL_SHADE_Plus:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()  # no fixed seed
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim

        # Population size (L-SHADE style reduction)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # Memory for successful parameters (SHADE)
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Sobol sequence initialization (space-filling)
        # Use random Sobol-like via LHS scramble
        from scipy.stats import qmc  # if available, else fallback to LHS
        try:
            sampler = qmc.Sobol(d=dim, scramble=True)
            lhs = sampler.random(pop_size)
        except:
            # fallback to Latin Hypercube
            lhs = np.random.rand(pop_size, dim)
            for j in range(dim):
                lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
        pop = lb + lhs * (ub - lb)

        # Evaluate initial population
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive for diversity (L-SHADE)
        archive = []
        archive_size = pop_size

        # Stagnation tracking
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Main loop
        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                # Keep archive within size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # Update pbest rate (time-dependent, L-SHADE style)
            p = 0.2 * (gen / max_gen) ** 2 + 0.1  # more aggressive early, then settle
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []

            # Compute current min fitness for weighting
            f_min = np.min(fitness)

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Choose pbest index (top p*pop_size individuals)
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Random selection from union of pop and archive (excluding current index)
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    def get_individual(idx):
                        if idx < pop_size:
                            return pop[idx]
                        else:
                            return archive[idx - pop_size]
                    x_r1 = get_individual(r1)
                    x_r2 = get_individual(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Draw F and CR from memory with Cauchy/normal perturbation
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation (current-to-pbest/1)
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Add parent to archive with diversity preservation: prefer replacing close archive individuals
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace the archive member closest to the new trial (distance-based)
                        dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                        idx_remove = np.argmin(dists)
                        archive[idx_remove] = pop[i].copy()

                    # Store successful parameters
                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))

                    # Update population
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update memory with weighted Lehmer mean (SHADE style)
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---- Local search: Nelder-Mead on best, triggered by stagnation or periodically ----
            nm_budget = int(0.15 * (self.budget - evals))
            if nm_budget > dim + 1 and (gen % 5 == 0 or stagnation_counter >= 3):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                step = (ub - lb) * 0.05
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = x_best
                for k in range(dim):
                    x = x_best.copy()
                    x[k] = np.clip(x[k] + step[k], lb[k], ub[k])
                    simplex[k+1] = x
                f_simplex = np.array([f_best] + [func(simplex[i]) for i in range(1, dim+1)])
                evals += dim
                nm_used = dim

                while nm_used < nm_budget and evals < self.budget:
                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]

                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = func(xr)
                    evals += 1; nm_used += 1

                    if fr < f_simplex[0]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe)
                        evals += 1; nm_used += 1
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
                            evals += 1; nm_used += 1
                            if fc < fr:
                                simplex[-1] = xc
                                f_simplex[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim+1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    f_simplex[i] = func(simplex[i])
                                    evals += 1; nm_used += 1
                        else:
                            xc = centroid - 0.5 * (centroid - simplex[-1])
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1; nm_used += 1
                            if fc < f_simplex[-1]:
                                simplex[-1] = xc
                                f_simplex[-1] = fc
                            else:
                                for i in range(1, dim+1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    f_simplex[i] = func(simplex[i])
                                    evals += 1; nm_used += 1

                    # Update global best
                    best_nm = np.argmin(f_simplex)
                    if f_simplex[best_nm] < self.f_opt:
                        self.f_opt = f_simplex[best_nm]
                        self.x_opt = simplex[best_nm].copy()

                # Inject best NM point into population if better than worst
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

            # Restart if severe stagnation
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                n_restart = max(1, int(0.5 * pop_size))
                # Use quasi-random restart around best (small perturbation) and also scattered points
                try:
                    sampler_restart = qmc.Sobol(d=dim, scramble=True)
                    sob = sampler_restart.random(n_restart)
                except:
                    sob = np.random.rand(n_restart, dim)
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # local perturbation
                        scale = 0.1 * (ub - lb) * (1 - gen / max_gen)
                        pop[idx] = best_copy + np.random.uniform(-1, 1, dim) * scale
                    else:
                        # scattered with Sobol
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                stagnation_counter = 0

        return self.f_opt, self.x_opt