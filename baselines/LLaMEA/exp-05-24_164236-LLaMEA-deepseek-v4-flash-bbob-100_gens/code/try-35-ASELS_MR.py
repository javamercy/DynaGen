import numpy as np

class ASELS_MR:
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

        # Population size (L-SHADE style, but exact from remaining budget)
        N_init = max(4, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_fes = self.budget
        fes_count = 0

        # SHADE memory
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
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            fes_count += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive for success parents
        archive = []
        archive_size = pop_size

        # Stagnation and diversity detection
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        last_improvement_gen = 0
        improved_in_last_few = False

        # L-BFGS memory
        L_mem = 8
        s_list = []
        y_list = []

        # Local search parameters
        ls_min_fes = dim * 10 + 50  # minimum evaluations before any local search
        ls_freq_init = max(8, dim // 2)
        ls_freq = ls_freq_init
        ls_max_iter = max(3, int(0.03 * self.budget / (2 * dim + 5)))
        ls_max_iter = min(ls_max_iter, 15)

        # Generation budget estimation
        def pop_size_next(fes_left):
            if fes_left < pop_size:
                return max(N_min, pop_size // 2)
            # linear reduction based on remaining evaluations
            ratio = fes_left / (self.budget - fes_count + fes_left + 1e-30)
            return max(N_min, int(N_init - (1 - ratio) * (N_init - N_min)))

        while fes_count < self.budget:
            gen += 1

            # Population reduction
            fes_left = self.budget - fes_count
            new_pop_size = pop_size_next(fes_left)
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate adaptive
            p = max(0.1, 0.25 * (gen / (self.budget / pop_size + 1)) ** 0.5 + 0.1)
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []

            for i in range(pop_size):
                if fes_count >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Generate r1, r2 from pop+archive
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    def get_ind(idx):
                        return pop[idx] if idx < pop_size else archive[idx - pop_size]
                    x_r1 = get_ind(r1)
                    x_r2 = get_ind(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F, CR from memory with Cauchy noise
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1 with archive
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
                fes_count += 1

                if f_trial <= fitness[i]:
                    # Archive management: keep last successful parent
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Random removal (L-SHADE style)
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
                        last_improvement_gen = gen

            # Update SHADE memory using weighted Lehmer mean
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # Detect stagnation and population diversity
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
                improved_in_last_few = True
            else:
                stagnation_counter += 1
                if gen - last_improvement_gen > 15:
                    improved_in_last_few = False

            # ----- Local search (L-BFGS) with efficient gradient -----
            ls_trigger = (fes_count > ls_min_fes and
                          (gen % ls_freq == 0 or
                           (np.std(fitness) < 0.5 and
                            stagnation_counter > 5 and
                            fes_left > dim * 10 + 20)))

            if ls_trigger:
                # Use best and few other promising points
                candidates = []
                sorted_idx = np.argsort(fitness)
                for k in range(min(3, pop_size)):
                    x0 = pop[sorted_idx[k]].copy()
                    f0 = fitness[sorted_idx[k]]
                    candidates.append((x0, f0))
                # Also always include current global best
                candidates = [(self.x_opt.copy(), self.f_opt)] + candidates[:2]

                for x0, f0 in candidates:
                    if fes_count + 2 * dim + 5 >= self.budget:
                        break
                    x = x0.copy()
                    f = f0
                    # One-sided gradient (dim+1 evaluations)
                    def grad_one_side(x):
                        g = np.zeros(dim)
                        fx = f
                        h = 1e-6 * (ub - lb) + 1e-8
                        for d in range(dim):
                            xp = x.copy()
                            xp[d] = np.clip(x[d] + h[d], lb[d], ub[d])
                            fp = func(xp)
                            g[d] = (fp - fx) / h[d]
                        fes_count += dim
                        return g
                    # Restart L-BFGS history for each candidate
                    s_list = []
                    y_list = []
                    for it in range(ls_max_iter):
                        if fes_count + dim + 3 >= self.budget:
                            break
                        g = grad_one_side(x)
                        if np.linalg.norm(g) < 1e-12:
                            break
                        # L-BFGS two-loop recursion
                        q = g.copy()
                        alphas = np.zeros(len(s_list))
                        for i in range(len(s_list)-1, -1, -1):
                            alphas[i] = np.dot(s_list[i], q) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                            q = q - alphas[i] * y_list[i]
                        d = -q.copy()
                        if len(s_list) > 0:
                            H0 = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-30)
                            d = H0 * d
                        for i in range(len(s_list)):
                            beta = np.dot(y_list[i], d) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                            d = d + (alphas[i] - beta) * s_list[i]
                        # Line search with polynomial interpolation
                        alpha_step = 1.0
                        c = 1e-4
                        fx = f
                        gx = g
                        for _ in range(10):
                            x_new = np.clip(x + alpha_step * d, lb, ub)
                            f_new = func(x_new)
                            fes_count += 1
                            if f_new <= fx + c * alpha_step * np.dot(gx, x_new - x):
                                break
                            # Quadratic interpolation for step size
                            if _ == 0 and f_new > fx:
                                a = np.dot(gx, d)
                                b = f_new - fx - a * alpha_step
                                if b > 0:
                                    alpha_step = np.clip( -a * alpha_step**2 / (2 * b), 0.1 * alpha_step, 0.9 * alpha_step)
                                else:
                                    alpha_step *= 0.5
                            else:
                                alpha_step *= 0.5
                        if alpha_step < 1e-12:
                            break
                        # Update L-BFGS history
                        s = x_new - x
                        # Compute new gradient at x_new (using two-sided to avoid re-evaluation)
                        y = g.copy()
                        # Actually we need gradient at x_new - recompute using one-sided (cost dim)
                        g_new = np.zeros(dim)
                        fx_new = f_new
                        for d in range(dim):
                            xp = x_new.copy()
                            xp[d] = np.clip(x_new[d] + 1e-6*(ub[d]-lb[d]), lb[d], ub[d])
                            fp = func(xp)
                            g_new[d] = (fp - fx_new) / (1e-6*(ub[d]-lb[d]))
                        fes_count += dim
                        y = g_new - g
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
                    # Inject candidate into population if better than worst
                    if f < fitness.max():
                        worst = np.argmax(fitness)
                        pop[worst] = x.copy()
                        fitness[worst] = f

            # ----- Restart (multi-cluster) -----
            if (stagnation_counter > max(15, int(0.1 * (self.budget / pop_size))) and
                fes_left > N_init * 10):
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Generate quasi-random points using Sobol (simulated)
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                # Fill population: half local, half global
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * (ub - lb) * (1 - gen / (self.budget / pop_size)) ** 2 + 0.02
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if fes_count < self.budget:
                        fitness[idx] = func(pop[idx])
                        fes_count += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory, archive, L-BFGS
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                last_improvement_gen = gen
                improved_in_last_few = True
                # Increase local search frequency after restart
                ls_freq = min(ls_freq + 2, max_gen // 4) if 'max_gen' in dir() else ls_freq + 2
                # Also increase population size if severely reduced
                if pop_size < N_init // 2:
                    pop_size = max(pop_size, N_init // 2)
                    # need to add new individuals
                    n_add = pop_size - len(fitness)
                    if n_add > 0 and fes_count < self.budget:
                        new_lhs = np.random.rand(n_add, dim)
                        for j in range(dim):
                            new_lhs[:, j] = (np.argsort(new_lhs[:, j]) + 0.5) / n_add
                        new_pop = lb + new_lhs * (ub - lb)
                        for j in range(n_add):
                            pop = np.vstack([pop, new_pop[j]])
                            fitness = np.append(fitness, func(new_pop[j]))
                            fes_count += 1
                            if fitness[-1] < self.f_opt:
                                self.f_opt = fitness[-1]
                                self.x_opt = new_pop[j].copy()

        return self.f_opt, self.x_opt