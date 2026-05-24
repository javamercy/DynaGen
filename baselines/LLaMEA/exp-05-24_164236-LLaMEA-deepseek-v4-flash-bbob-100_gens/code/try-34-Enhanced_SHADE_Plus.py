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
        span = ub - lb

        # Population size (L‑SHADE style, with slower reduction)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.2)   # slightly more generations

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Quasi‑random Sobol initialisation (Latin Hypercube style)
        # Use low‑discrepancy pattern
        sob = np.random.rand(pop_size, dim)
        for j in range(dim):
            sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / pop_size
        pop = lb + sob * span

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive (L‑SHADE style)
        archive = []
        archive_size = pop_size

        # Stagnation tracking
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters
        ls_freq_init = max(12, int(0.06 * max_gen))
        ls_freq = ls_freq_init
        ls_max_iter = max(4, int(0.04 * (self.budget / (dim + 5))))
        ls_max_iter = min(ls_max_iter, 12)

        # L‑BFGS memory
        L_mem = 5
        s_list = []
        y_list = []

        # Nelder‑Mead parameters (fast restart helper)
        nm_max_evals = max(3 * dim, 30)

        while evals < self.budget:
            gen += 1

            # Linear population size reduction (slower)
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Adaptive pbest rate
            p = 0.2 * (gen / max_gen) ** 2 + 0.1
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

                # Select r1, r2 from pop and archive (better diversity)
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

                # Sample F and CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current‑to‑pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial (60%) or exponential (40%)
                trial = np.zeros(dim)
                if np.random.rand() < 0.6:
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

                # Reflective boundary handling (better than clamping)
                # Reflect points outside bounds back inside
                out_low = trial < lb
                out_high = trial > ub
                trial[out_low] = 2 * lb[out_low] - trial[out_low]
                trial[out_high] = 2 * ub[out_high] - trial[out_high]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
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

            # Update SHADE memory (weighted Lehmer mean for F)
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---------- Local search trigger ----------
            if (gen % ls_freq == 0 and 
                (self.budget - evals) > dim * 6 + 30 and
                np.std(fitness) < 1.5 and 
                stagnation_counter > 3):

                # Use L‑BFGS only if enough budget left for gradient (2*dim per gradient)
                if (self.budget - evals) > (dim * 2 * ls_max_iter + 30):
                    h = 1e-5 * span + 1e-8
                    def grad(x):
                        g = np.zeros(dim)
                        for d in range(dim):
                            xp = np.clip(x + np.eye(1,dim,d) * h[d], lb, ub)[0]
                            xn = np.clip(x - np.eye(1,dim,d) * h[d], lb, ub)[0]
                            g[d] = (func(xp) - func(xn)) / (2 * h[d])
                        return g

                    x = self.x_opt.copy()
                    f = self.f_opt
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
                            denom = np.dot(y_list[i], s_list[i]) + 1e-30
                            alpha[i] = np.dot(s_list[i], q) / denom
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
                    # Inject best found into population
                    if self.f_opt < fitness.max():
                        worst = np.argmax(fitness)
                        pop[worst] = self.x_opt.copy()
                        fitness[worst] = self.f_opt
                else:
                    # Cheap Nelder‑Mead on best point (gradient‑free)
                    # Works well on many BBOB functions
                    nm_evals = 0
                    # Build simplex around best point
                    simplex = np.zeros((dim + 1, dim))
                    simplex[0] = self.x_opt
                    for j in range(dim):
                        delta = 0.05 * span[j]
                        point = self.x_opt.copy()
                        point[j] = np.clip(point[j] + delta, lb[j], ub[j])
                        simplex[j+1] = point
                    f_simplex = np.array([func(simplex[i]) for i in range(dim+1)])
                    evals += dim + 1
                    nm_evals += dim + 1
                    # Sort
                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]
                    if f_simplex[0] < self.f_opt:
                        self.f_opt = f_simplex[0]
                        self.x_opt = simplex[0].copy()
                    # Simple Nelder‑Mead iterations (without full expansion/shrink to save budget)
                    while nm_evals < nm_max_evals and evals < self.budget:
                        centroid = np.mean(simplex[:dim], axis=0)
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
                        else:
                            worst_idx = dim  # actually the last is worst
                            # Look at second worst? simplified: if fr < f_simplex[-2]? Not needed, use standard.
                            # If reflection is better than second worst, accept.
                            if fr < f_simplex[dim-1]:
                                simplex[-1] = xr
                                f_simplex[-1] = fr
                            else:
                                # contraction
                                xc = centroid + 0.5 * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc)
                                evals += 1
                                nm_evals += 1
                                if fc < f_simplex[-1]:
                                    simplex[-1] = xc
                                    f_simplex[-1] = fc
                                else:
                                    # shrink (keep only best half)
                                    for i in range(1, dim+1):
                                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                        f_simplex[i] = func(simplex[i])
                                        evals += 1
                                        nm_evals += 1
                        # Re‑sort
                        order = np.argsort(f_simplex)
                        simplex = simplex[order]
                        f_simplex = f_simplex[order]
                        if f_simplex[0] < self.f_opt:
                            self.f_opt = f_simplex[0]
                            self.x_opt = simplex[0].copy()
                    # Inject best simplex point into population
                    if self.f_opt < fitness.max():
                        worst = np.argmax(fitness)
                        pop[worst] = self.x_opt.copy()
                        fitness[worst] = self.f_opt

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.08 * max_gen)):
                n_restart = max(1, int(0.5 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Generate quasi‑random Sobol points
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.05 * span * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + sob[idx] * span
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory, archive, L‑BFGS history
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = min(max_gen // 4, ls_freq + 2)

        return self.f_opt, self.x_opt