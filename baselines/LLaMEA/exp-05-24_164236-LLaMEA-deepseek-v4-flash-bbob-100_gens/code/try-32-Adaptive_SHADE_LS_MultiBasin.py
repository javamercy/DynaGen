import numpy as np

class Adaptive_SHADE_LS_MultiBasin:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        dim = self.dim
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # L-SHADE population size reduction
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory for F and CR
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube initial population
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

        # Archive for L-SHADE
        archive = []
        archive_size = pop_size

        # Multi-basin elite archive (store best distinct solutions)
        basin_list = []            # list of (x, f) pairs
        BASIN_MAX = 5
        basin_dist_thresh = 0.2 * (ub - lb)  # distance threshold for distinct basins

        # Stagnation tracking
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters – adaptive frequency
        ls_freq = max(10, int(0.08 * max_gen))
        ls_success_streak = 0   # track consecutive successful LS
        ls_fail_streak = 0
        ls_max_iter = max(3, int(0.05 * (self.budget / (2*dim + 5))))
        ls_max_iter = min(ls_max_iter, 15)
        ls_budget_fraction = 0.08

        # L-BFGS memory
        L_mem = 5
        s_list = []
        y_list = []

        # Main loop
        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]].copy()
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # p-best rate (time-dependent)
            p = 0.2 * (1 - evals / self.budget) + 0.1
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Select r1, r2 from pop and archive
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    def get_ind(idx):
                        if idx < pop_size:
                            return pop[idx]
                        else:
                            return archive[idx - pop_size]
                    x_r1 = get_ind(r1)
                    x_r2 = get_ind(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F, CR from memory with noise
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial (70%) or exponential (30%)
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
                    # Update archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                        idx_remove = np.argmin(dists)
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

            # Update SHADE memory
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---------- Adaptive Local Search (L-BFGS) ----------
            # Trigger based on budgets, diversity, and stagnation
            if (gen % ls_freq == 0 and 
                (self.budget - evals) > dim * 5 + 20 and
                np.std(fitness) < 1.0 and 
                stagnation_counter > 3):
                h = 1e-5 * (ub - lb) + 1e-8
                def grad(x):
                    g = np.zeros(dim)
                    for d in range(dim):
                        xp = np.clip(x + np.eye(1,dim,d) * h[d], lb, ub)[0]
                        xn = np.clip(x - np.eye(1,dim,d) * h[d], lb, ub)[0]
                        g[d] = (func(xp) - func(xn)) / (2 * h[d])
                    return g

                x = self.x_opt.copy()
                f = self.f_opt
                ls_improved = False
                for it in range(ls_max_iter):
                    if evals + 2*dim >= self.budget:
                        break
                    g = grad(x)
                    evals += 2*dim
                    if np.linalg.norm(g) < 1e-12:
                        break
                    # L-BFGS two-loop recursion
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        alpha[i] = np.dot(s_list[i], q) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        q = q - alpha[i] * y_list[i]
                    d = -q
                    if len(s_list) > 0:
                        H0 = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-30)
                        d = H0 * d
                    for i in range(len(s_list)):
                        beta = np.dot(y_list[i], d) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        d = d + (alpha[i] - beta) * s_list[i]
                    # Line search (Armijo)
                    alpha_step = 1.0
                    c = 1e-4
                    fx = f
                    for _ in range(12):
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= fx + c * alpha_step * np.dot(g, x_new - x):
                            break
                        alpha_step *= 0.5
                    if alpha_step < 1e-12:
                        break
                    # Update L-BFGS history
                    s = x_new - x
                    y = grad(x_new) - g
                    evals += 2*dim
                    sy = np.dot(s, y)
                    if sy > 1e-10:
                        if len(s_list) >= L_mem:
                            s_list.pop(0)
                            y_list.pop(0)
                        s_list.append(s.copy())
                        y_list.append(y.copy())
                    x = x_new
                    f = f_new
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()
                        ls_improved = True
                # Inject best local point into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                # Update LS success streak
                if ls_improved:
                    ls_success_streak += 1
                    ls_fail_streak = 0
                    # If LS successful, do it more often
                    ls_freq = max(5, int(ls_freq * 0.8))
                else:
                    ls_fail_streak += 1
                    ls_success_streak = 0
                    # If LS fails repeatedly, reduce frequency
                    if ls_fail_streak >= 3:
                        ls_freq = min(max_gen // 4, int(ls_freq * 1.5))

            # ---------- Stagnation Detection & Multi-Basin Restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
                # Update basin list with new best if distinct
                new_basin = True
                for b in basin_list:
                    if np.linalg.norm(self.x_opt - b[0]) < basin_dist_thresh:
                        new_basin = False
                        break
                if new_basin:
                    basin_list.append((self.x_opt.copy(), self.f_opt))
                    if len(basin_list) > BASIN_MAX:
                        # Keep the best ones (by fitness)
                        basin_list.sort(key=lambda x: x[1])
                        basin_list = basin_list[:BASIN_MAX]
            else:
                stagnation_counter += 1

            if stagnation_counter > max(12, int(0.1 * max_gen)):
                # Multi-basin restart: generate subpopulations around each basin + global
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Ensure we have at least one basin
                if len(basin_list) == 0:
                    basin_list = [(best_copy, best_f)]
                # Determine how many individuals per basin
                n_basins = len(basin_list)
                per_basin = n_restart // (n_basins + 1)  # +1 for global random
                global_count = n_restart - per_basin * n_basins
                if per_basin < 1:
                    per_basin = 1
                    global_count = 0
                new_pop = []
                new_fit = []
                # For each basin, sample LHS around it
                for (bx, bf) in basin_list:
                    for _ in range(per_basin):
                        # Perturbation scale decreases with generation
                        sigma = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        x_new = bx + np.random.randn(dim) * sigma
                        x_new = np.clip(x_new, lb, ub)
                        new_pop.append(x_new)
                        if evals < self.budget:
                            f_new = func(x_new)
                            evals += 1
                            new_fit.append(f_new)
                            if f_new < self.f_opt:
                                self.f_opt = f_new
                                self.x_opt = x_new.copy()
                        else:
                            break
                # Add global random individuals using LHS
                if global_count > 0 and evals < self.budget:
                    sob = np.random.rand(global_count, dim)
                    for j in range(dim):
                        sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / global_count
                    for idx in range(global_count):
                        x_new = lb + sob[idx] * (ub - lb)
                        new_pop.append(x_new)
                        if evals < self.budget:
                            f_new = func(x_new)
                            evals += 1
                            new_fit.append(f_new)
                            if f_new < self.f_opt:
                                self.f_opt = f_new
                                self.x_opt = x_new.copy()
                        else:
                            break
                # Replace population
                if len(new_pop) >= N_min:
                    pop = np.array(new_pop[:pop_size])
                    fitness = np.array(new_fit[:pop_size])
                    pop_size = len(pop)
                # Reset memory, archive, L-BFGS history
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                # Increase LS frequency after restart 
                ls_freq = max(10, int(ls_freq * 0.9))

            # Catapult mechanism: if still stagnation after restart, apply large perturbation to best
            if stagnation_counter > max(20, int(0.15 * max_gen)):
                # Perturb best solution by random direction scaled by range
                perturbation = np.random.uniform(-0.3, 0.3, dim) * (ub - lb)
                x_new = np.clip(self.x_opt + perturbation, lb, ub)
                if evals < self.budget:
                    f_new = func(x_new)
                    evals += 1
                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = x_new.copy()
                    # Inject into population if worse?
                stagnation_counter = 0

        return self.f_opt, self.x_opt