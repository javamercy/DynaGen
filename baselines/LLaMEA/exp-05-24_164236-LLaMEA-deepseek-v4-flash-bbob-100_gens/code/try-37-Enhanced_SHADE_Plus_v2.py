import numpy as np
import math

class Enhanced_SHADE_Plus_v2:
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

        # Population size initialization (L-SHADE style)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory for F and CR
        mem_size = 10
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling (LHS) initial population
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

        # Stagnation detection
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters - more adaptive
        ls_freq_init = max(10, int(0.06 * max_gen))
        ls_freq = ls_freq_init
        ls_max_iter = max(5, int(0.04 * (self.budget / (2*dim + 5))))
        ls_max_iter = min(ls_max_iter, 20)
        ls_budget_fraction = 0.06

        # Nelder-Mead parameters (when L-BFGS is too expensive)
        nm_max_iters = max(5, int(0.02 * (self.budget / (dim+2))))

        # L-BFGS memory
        L_mem = 10
        s_list = []
        y_list = []

        # Adaptive pbest rate
        pbest_rate = 0.2  # initial
        pbest_success = []

        # Success-history for convergence detection
        recent_improvements = []

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
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Adaptive pbest rate based on success history
            if len(pbest_success) > 10:
                suc_rate = np.mean(pbest_success[-10:])
                pbest_rate = 0.1 + 0.4 * (1 - suc_rate)  # low success -> higher pbest (more exploration)
            else:
                pbest_rate = 0.2
            pbest_rate = np.clip(pbest_rate, 0.05, 0.6)

            # Mutation scaling factor for current-to-pbest
            p = pbest_rate

            success_F = []
            success_CR = []
            weight = []

            # Track number of successful mutations for this generation
            gen_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
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

                # Sample F, CR from memory (with noise)
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1 (with occasional current-to-rand/1 for diversity)
                if np.random.rand() < 0.85:  # 85% standard current-to-pbest
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                else:  # 15% current-to-rand/1 (rotation-invariant)
                    mutant = pop[i] + F * (x_r1 - x_r2) + F * (np.random.rand(dim) * 2 - 1) * (ub - lb) * 0.1
                    # Add small random perturbation to avoid stagnation

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
                        # Replace worst archive member by distance to pop[i]
                        if len(archive) > 0:
                            dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                            idx_remove = np.argmin(dists)
                            archive[idx_remove] = pop[i].copy()
                        else:
                            archive.append(pop[i].copy())

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))
                    gen_success += 1

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

            # Track success rate for pbest adaptation
            pbest_success.append(gen_success / max(1, pop_size))

            # -------- Adaptive Local Search (L-BFGS) ----------
            # Criteria: enough budget, generation frequency, and stagnation indicator
            if (gen % ls_freq == 0 and 
                (self.budget - evals) > dim * 5 + 30 and
                stagnation_counter > 2):

                # Finite difference gradient (2-sided)
                h = 1e-7 * (ub - lb) + 1e-8
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
                    # Compute search direction via L-BFGS
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
                # Inject best local point into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt

                # If L-BFGS used many evals but no improvement, try simple Nelder-Mead on best point
                if f >= self.f_opt and evals < self.budget:
                    # Simple Nelder-Mead with small simplex
                    nm_simplex = [self.x_opt.copy()]
                    for d_ in range(dim):
                        p = self.x_opt.copy()
                        p[d_] += 0.01 * (ub[d_] - lb[d_])
                        p = np.clip(p, lb, ub)
                        nm_simplex.append(p)
                    nm_fvals = np.array([func(p) for p in nm_simplex])
                    evals += len(nm_simplex)
                    nm_iters = 0
                    while nm_iters < nm_max_iters and evals < self.budget:
                        # Sort simplex
                        idx_sorted = np.argsort(nm_fvals)
                        nm_simplex = [nm_simplex[i] for i in idx_sorted]
                        nm_fvals = nm_fvals[idx_sorted]
                        centroid = np.mean(nm_simplex[:-1], axis=0)
                        # Reflection
                        xr = np.clip(centroid + 0.9 * (centroid - nm_simplex[-1]), lb, ub)
                        fr = func(xr)
                        evals += 1
                        if nm_fvals[0] <= fr < nm_fvals[-2]:
                            nm_simplex[-1] = xr
                            nm_fvals[-1] = fr
                        elif fr < nm_fvals[0]:
                            # Expansion
                            xe = np.clip(centroid + 1.3 * (xr - centroid), lb, ub)
                            fe = func(xe)
                            evals += 1
                            if fe < fr:
                                nm_simplex[-1] = xe
                                nm_fvals[-1] = fe
                            else:
                                nm_simplex[-1] = xr
                                nm_fvals[-1] = fr
                        else:
                            # Contraction
                            xc = np.clip(centroid + 0.4 * (nm_simplex[-1] - centroid), lb, ub)
                            fc = func(xc)
                            evals += 1
                            if fc < nm_fvals[-1]:
                                nm_simplex[-1] = xc
                                nm_fvals[-1] = fc
                            else:
                                # Shrink
                                for i_ in range(1, len(nm_simplex)):
                                    nm_simplex[i_] = nm_simplex[0] + 0.5 * (nm_simplex[i_] - nm_simplex[0])
                                    nm_simplex[i_] = np.clip(nm_simplex[i_], lb, ub)
                                    nm_fvals[i_] = func(nm_simplex[i_])
                                evals += len(nm_simplex)-1
                        nm_iters += 1
                    # Update best
                    best_idx = np.argmin(nm_fvals)
                    if nm_fvals[best_idx] < self.f_opt:
                        self.f_opt = nm_fvals[best_idx]
                        self.x_opt = nm_simplex[best_idx].copy()
                        # Inject into population
                        worst_pop = np.argmax(fitness)
                        pop[worst_pop] = self.x_opt.copy()
                        fitness[worst_pop] = self.f_opt

            # -------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.08 * max_gen)):
                # Restart: generate new subpopulation using Sobol sequences around best + global sampling
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Generate Sobol-like quasi-random points (simple implementation: shuffled LHS)
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # Local perturbation around best
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
                        # Global LHS points
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
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
                # Adjust local search frequency to avoid frequent restarts
                ls_freq = min(max_gen // 4, ls_freq + 2)

        return self.f_opt, self.x_opt