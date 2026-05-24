import numpy as np

class Enhanced_HybridL_SHADE_Plus_V3:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim

        # Population initial size (L-SHADE style reduction)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling for initial population
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

        # L-SHADE archive
        archive = []
        archive_size = pop_size

        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # pbest ratio (increases over time)
            p = 0.2 * (gen / max_gen) ** 2 + 0.1
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Choose r1, r2 from union of pop and archive (excluding i)
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

                # Sample F and CR from memory with Cauchy-like perturbation
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial with exponential mixture
                trial = np.zeros(dim)
                if np.random.rand() < 0.7:
                    j_rand = np.random.randint(dim)
                    mask = np.random.rand(dim) < CR
                    mask[j_rand] = True
                    trial = np.where(mask, mutant, pop[i])
                else:
                    start = np.random.randint(dim)
                    L = 0
                    while L < dim and np.random.rand() < CR:
                        L += 1
                    indices = (np.arange(dim) + start) % dim
                    mask = np.zeros(dim, dtype=bool)
                    mask[indices[:L]] = True
                    trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Add parent to archive (replace oldest if full)
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace a random archive member to maintain diversity
                        idx_remove = np.random.randint(len(archive))
                        archive[idx_remove] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update SHADE memory with weighted Lehmer mean for F, arithmetic for CR
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---- Local search: directional refinement with golden-section line search ----
            ls_budget = int(0.10 * (self.budget - evals))
            if ls_budget > dim and (gen % 4 == 0 or stagnation_counter >= 4):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                ndirs = min(dim, ls_budget // 3)  # number of random directions to try
                ls_used = 0

                for _ in range(ndirs):
                    if ls_used >= ls_budget or evals >= self.budget:
                        break
                    # Random unit direction
                    d = np.random.randn(dim)
                    d = d / (np.linalg.norm(d) + 1e-30)
                    # Golden-section line search from x_best along d (bounded)
                    a = 0.0
                    b = 2.0  # initial step bracket (scaled by domain size)
                    scale = np.linalg.norm(ub - lb) * 0.1  # domain size factor
                    phi = (np.sqrt(5) - 1) / 2.0  # golden ratio

                    # Evaluate at two initial points to bracket minimum
                    x1 = np.clip(x_best + scale * a * d, lb, ub)
                    f1 = func(x1) if a == 0.0 else f_best  # reuse best fitness
                    x2 = np.clip(x_best + scale * b * d, lb, ub)
                    f2 = func(x2)
                    evals += 1; ls_used += 1
                    if np.allclose(x1, x2):
                        continue

                    # Simple golden-section for a few steps
                    for _ in range(min(6, ls_budget - ls_used)):
                        if evals >= self.budget:
                            break
                        c = b - phi * (b - a)
                        xc = np.clip(x_best + scale * c * d, lb, ub)
                        fc = func(xc)
                        evals += 1; ls_used += 1
                        d_ = a + phi * (b - a)
                        xd = np.clip(x_best + scale * d_ * d, lb, ub)
                        fd = func(xd)
                        evals += 1; ls_used += 1
                        if fc < fd:
                            b = d_
                            # keep c as new point? simplified: keep best found
                        else:
                            a = c
                        # Update best in this line search
                        best_x = x_best + scale * ((a + b) / 2) * d
                        best_x = np.clip(best_x, lb, ub)
                        f_best_local = func(best_x)
                        evals += 1; ls_used += 1
                        if f_best_local < self.f_opt:
                            self.f_opt = f_best_local
                            self.x_opt = best_x.copy()
                            x_best = self.x_opt.copy()
                            f_best = self.f_opt

                # Inject best into population if better than worst
                worst_idx = np.argmax(fitness)
                if self.f_opt < fitness[worst_idx]:
                    pop[worst_idx] = self.x_opt.copy()
                    fitness[worst_idx] = self.f_opt

            # Stagnation detection and restart
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.08 * max_gen)):
                # Diversity restoration: replace 60% of population
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                # Generate Sobol-like sequence using Latin Hypercube (simple)
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
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