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
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)

        # population size (L-SHADE style)
        N_init = max(8, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # initial population (Latin Hypercube)
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

        # archive for L-SHADE
        archive = []
        archive_size = pop_size

        # stagnation tracking
        best_old = self.f_opt
        stagnation = 0

        # local search parameters
        ls_freq = max(10, int(0.08 * max_gen))
        ls_max_iter = max(3, int(0.05 * (self.budget / (2*dim + 5))))
        ls_max_iter = min(ls_max_iter, 15)

        # L-BFGS history
        L_mem = 5
        s_list = []
        y_list = []

        # diversity metric (average pairwise distance)
        def diversity(pop):
            if len(pop) <= 1:
                return 0.0
            mean = np.mean(pop, axis=0)
            return np.mean(np.linalg.norm(pop - mean, axis=1))

        gen = 0
        while evals < self.budget:
            gen += 1

            # linear population reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx = np.argsort(fitness)
                pop = pop[idx[:new_pop_size]].copy()
                fitness = fitness[idx[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate (adaptive based on diversity)
            div = diversity(pop)
            p = 0.2 * (gen / max_gen) ** 2 + 0.1 * min(div, 2.0) / 2.0
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

                # select r1, r2 from pop+archive
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

                # sample F, CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # crossover (binomial)
                trial = np.copy(pop[i])
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # archive update
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

            # update SHADE memory
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---------- Local search (L-BFGS) with adaptive trigger ----------
            # trigger if stagnation and enough budget, or every ls_freq generations
            do_ls = (gen % ls_freq == 0) or (stagnation > 4 and (self.budget - evals) > dim * 5 + 20)
            if do_ls and (self.budget - evals) > dim * 5 + 20:
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
                # L-BFGS two-loop recursion
                for it in range(ls_max_iter):
                    if evals + 2*dim >= self.budget:
                        break
                    g = grad(x)
                    evals += 2*dim
                    if np.linalg.norm(g) < 1e-12:
                        break
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
                    # Armijo line search
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
                # inject best local point into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > max(8, int(0.08 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # generate new points: half near best (opposition), half global
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # opposition-based local: x_best + small perturbation
                        pert = np.random.randn(dim) * 0.02 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        new_x = best_copy + pert
                    else:
                        # global random (Sobol-like via LHS)
                        r = np.random.rand(dim)
                        r = (np.argsort(r) + 0.5) / dim
                        new_x = lb + r * (ub - lb)
                    new_x = np.clip(new_x, lb, ub)
                    if evals < self.budget:
                        f_new = func(new_x)
                        evals += 1
                        if f_new < self.f_opt:
                            self.f_opt = f_new
                            self.x_opt = new_x.copy()
                        # replace worst in population
                        if idx < pop_size:
                            worst_idx = np.argmax(fitness)
                            if f_new < fitness[worst_idx]:
                                pop[worst_idx] = new_x.copy()
                                fitness[worst_idx] = f_new
                # reset memory, archive, L-BFGS history
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation = 0
                # reduce ls_freq to avoid frequent local search after restart
                ls_freq = min(max_gen // 3, ls_freq + 1)

        return self.f_opt, self.x_opt