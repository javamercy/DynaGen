import numpy as np

class Enhanced_SHADE_Plus:
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

        # Population size – exponential reduction from N_init to N_min
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

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
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        archive = []
        archive_size = 2 * pop_size  # larger archive

        # Stagnation and restart
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters (adaptive)
        ls_freq_init = max(5, int(0.05 * max_gen))
        ls_freq = ls_freq_init
        ls_max_iter = max(3, int(0.03 * (self.budget / (2*dim + 5))))
        ls_max_iter = min(ls_max_iter, 12)
        L_mem = 5
        s_list = []
        y_list = []
        ls_success_counter = 0  # success streak for LS

        while evals < self.budget:
            gen += 1

            # Exponential population reduction
            new_pop_size = max(N_min, int(N_init * (1 - gen / max_gen) ** 1.5))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Adaptive pbest rate (decreasing over generations)
            p = 0.15 + 0.15 * (1 - gen / max_gen) ** 2
            p = min(p, 0.4)

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

                # Select r1, r2 from population and archive
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

                # Sample F, CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial with occasional exponential
                trial = np.zeros(dim)
                if np.random.rand() < 0.8:
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
                    # Archive insertion: replace farthest by distance-weighted fitness
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # remove the one most similar to pop[i] in terms of function + distance
                        dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                        # Weight with fitness difference to avoid removing very good points
                        fit_diffs = np.array([abs(f - f_trial) for f in archive]) + 1e-12
                        scores = dists / (fit_diffs + 1e-12)
                        idx_remove = np.argmin(scores)
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

            # ---------- Adaptive L-BFGS local search ----------
            # Trigger less often when success streak high, more often when stagnating
            if (gen % ls_freq == 0 and self.budget - evals > dim * 8 + 20):
                # Heuristic: check whether the best point has changed recently
                if stagnation_counter < 5:
                    # If not stagnating, skip to save budget
                    pass
                else:
                    # Use best point for LS
                    x = self.x_opt.copy()
                    f = self.f_opt

                    # Gradient via central differences (2*dim evaluations)
                    h = 1e-5 * (ub - lb) + 1e-8
                    def grad(x):
                        g = np.zeros(dim)
                        for d in range(dim):
                            xp = np.clip(x + np.eye(1,dim,d) * h[d], lb, ub)[0]
                            xn = np.clip(x - np.eye(1,dim,d) * h[d], lb, ub)[0]
                            g[d] = (func(xp) - func(xn)) / (2 * h[d])
                        return g

                    g = grad(x)
                    evals += 2 * dim
                    if np.linalg.norm(g) < 1e-12:
                        continue

                    # Two-loop recursion for L-BFGS direction
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        rho = 1.0 / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        alpha[i] = rho * np.dot(s_list[i], q)
                        q = q - alpha[i] * y_list[i]
                    d = -q
                    if len(s_list) > 0:
                        H0 = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-30)
                        d = H0 * d
                    for i in range(len(s_list)):
                        rho = 1.0 / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        beta = rho * np.dot(y_list[i], d)
                        d = d + (alpha[i] - beta) * s_list[i]

                    # Armijo line search
                    alpha_step = 1.0
                    c = 1e-4
                    fx = f
                    found = False
                    for _ in range(10):
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= fx + c * alpha_step * np.dot(g, x_new - x):
                            found = True
                            break
                        alpha_step *= 0.5
                    if not found or alpha_step < 1e-12:
                        # fallback: simple gradient descent
                        alpha_step = min(1.0, 0.5 * max(ub - lb) / (np.linalg.norm(g)+1e-12))
                        x_new = np.clip(x - alpha_step * g, lb, ub)
                        f_new = func(x_new)
                        evals += 1

                    # Update L-BFGS history
                    s = x_new - x
                    if np.dot(s, g) - f_new + fx > 1e-10:  # curvature condition
                        y = grad(x_new) - g
                        evals += 2 * dim
                        sy = np.dot(s, y)
                        if sy > 1e-10:
                            if len(s_list) >= L_mem:
                                s_list.pop(0)
                                y_list.pop(0)
                            s_list.append(s.copy())
                            y_list.append(y.copy())
                            ls_success_counter += 1
                        else:
                            ls_success_counter = max(0, ls_success_counter - 1)
                    else:
                        ls_success_counter = max(0, ls_success_counter - 1)
                    # Update best if improved
                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = x_new.copy()
                    # Inject into population
                    if self.f_opt < fitness.max():
                        worst = np.argmax(fitness)
                        pop[worst] = self.x_opt.copy()
                        fitness[worst] = self.f_opt

            # ---------- Stagnation detection and Cauchy restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.06 * max_gen)):
                n_restart = max(2, int(0.5 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Generate new points: half around best with Cauchy tails, half quasi-random
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # Cauchy distribution: heavy tails for exploration
                        scale = 0.1 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + scale * np.random.standard_cauchy(dim)
                    else:
                        # Quasi-random LHS
                        sob = np.random.rand(dim)
                        sob = (np.argsort(sob) + 0.5) / dim
                        pop[idx] = lb + sob * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Fill remaining population if n_restart < pop_size
                for idx in range(n_restart, pop_size):
                    pop[idx] = lb + np.random.rand(dim) * (ub - lb)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory, archive, L-BFGS history
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_success_counter = 0
                # Increase LS interval to avoid premature restart spamming
                ls_freq = min(max_gen // 3, ls_freq + 3)

        return self.f_opt, self.x_opt