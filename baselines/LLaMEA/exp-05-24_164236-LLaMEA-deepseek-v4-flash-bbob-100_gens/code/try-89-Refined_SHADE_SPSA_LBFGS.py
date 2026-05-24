import numpy as np

class Refined_SHADE_SPSA_LBFGS:
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
        domain_range = ub - lb

        # ---------- initialization: Sobol-like LHS ----------
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)

        # Sobol low-discrepancy sequence via bitwise generation (simple version)
        def sobol_sample(n, d):
            # Use primitive polynomials for up to d=40
            from math import log2
            poly = [1,1,2,1,2,3,1,3,4,2,5,3,4,5,6,1,6,7,4,8,5,7,9,6,8,10,7,9,11,8,10,12,9,11,13,10,12,14,11,13,15,12,14,16]
            # initialize direction vectors
            max_n = n
            bits = int(log2(max_n)) + 2
            prim = [1,3,3,7,9,11,13,15,19,21,23,27,29,31,33,35,43,47,49,55]
            dirs = np.zeros((d, bits), dtype=int)
            for i in range(d):
                p = poly[i] if i < len(poly) else 1
                m = [1]
                while len(m) < bits:
                    m.append((p>>(len(m)-1)) ^ (m[-1]*2))
                # convert to fractions
                for j in range(bits):
                    dirs[i,j] = m[j] * (2**bits)
            samp = np.zeros((n, d))
            for i in range(n):
                prev = 0
                gray = i ^ (i>>1)
                for j in range(d):
                    val = 0
                    for k in range(bits):
                        if (gray >> k) & 1:
                            val ^= dirs[j,k]
                    samp[i,j] = val / (2**bits)
            return samp

        n_sob = pop_size
        # Use Sobol if dim <= 16 (simple implementation) else LHS
        use_sobol = dim <= 16
        if use_sobol:
            try:
                sobol = sobol_sample(n_sob, dim)  # custom, may not be perfect but good enough
            except Exception:
                use_sobol = False
        if not use_sobol:
            perm = np.tile(np.arange(1, n_sob + 1), (dim, 1)).T
            for j in range(dim):
                perm[:, j] = np.random.permutation(perm[:, j])
            sobol = (perm - 0.5) / n_sob
        pop = lb + sobol * domain_range

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Archive
        archive = []
        archive_size = pop_size

        # Stagnation and local search control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(8, int(0.05 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))

        # L-BFGS memory
        L_mem = 5
        s_list = []
        y_list = []

        # Nelder-Mead memory (for second local search)
        nm_memory = None  # simplex for NM

        # Success rates
        success_rates = []

        # Helper: robust gradient estimation using average of 2 random directions
        def robust_spsa_grad(x, num_dirs=1):
            # SPSA with optional averaging over num_dirs directions (cost 2*num_dirs)
            c = 1e-3 * domain_range.mean()
            grad = np.zeros(dim)
            f_vals = []
            for _ in range(num_dirs):
                delta = np.random.choice([-1, 1], size=dim)
                x_plus = np.clip(x + c * delta, lb, ub)
                x_minus = np.clip(x - c * delta, lb, ub)
                fp = func(x_plus)
                fm = func(x_minus)
                if np.isinf(fp) or np.isinf(fm):
                    return None, None, None
                f_vals.extend([fp, fm])
                grad += (fp - fm) / (2 * c) * delta
            grad /= num_dirs
            # return grad and the two function values of the last pair? Use median f for line search
            f0 = np.median(f_vals)
            return grad, f0, f0

        # Nelder-Mead local search (lightweight, 2+dim evaluations)
        def nelder_mead_step(center, f_center, func):
            # Build simplex: center + scaled unit simplex
            n = dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            # create simplex
            simplex = np.zeros((n+1, n))
            fvals = np.zeros(n+1)
            simplex[0] = center
            fvals[0] = f_center
            for i in range(n):
                dx = np.zeros(n)
                dx[i] = 0.05 * domain_range[i]
                simplex[i+1] = np.clip(center + dx, lb, ub)
                fvals[i+1] = func(simplex[i+1])
            evals_local = n
            # one iteration of Nelder-Mead
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            centroid = np.mean(simplex[:-1], axis=0)
            # reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evals_local += 1
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lb, ub)
                fe = func(xe)
                evals_local += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evals_local += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lb, ub)
                        fvals[i] = func(simplex[i])
                        evals_local += 1
            # return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx], evals_local

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction (exponential-like)
            ratio = max(0, 1 - (gen / max_gen) ** 1.2)
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            p = 0.1 + 0.4 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                union = list(range(pop_size)) + list(range(len(archive)))
                try:
                    union.remove(i)
                except ValueError:
                    pass
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

                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1 with additional archive perturbation
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Binomial crossover (70%) or exponential (30%)
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
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx_remove = np.random.randint(len(archive))
                        archive[idx_remove] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))
                    n_success += 1

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update SHADE memory
            if len(success_F) > 0:
                w = np.array(weight)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                    CR_mean = np.sum(w * np.array(success_CR))
                    mem_F[mem_idx] = F_lehmer
                    mem_CR[mem_idx] = CR_mean
                    mem_idx = (mem_idx + 1) % mem_size

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Local search (hybrid) ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            trigger_ls = (gen % ls_freq == 0 and budget_left > 30 and diversity and low_success)

            if trigger_ls:
                x = self.x_opt.copy()
                f = self.f_opt

                # --- Attempt SPSA-LBFGS first (if gradient reliable) ---
                # Check gradient magnitude with a quick SPSA evaluation
                g_test, _, _ = robust_spsa_grad(x, num_dirs=1)
                if g_test is not None and np.linalg.norm(g_test) > 1e-12:
                    # do L-BFGS
                    ls_iters = max(2, min(10, int(0.03 * budget_left / dim)))
                    for it in range(ls_iters):
                        if evals + 3 >= self.budget:
                            break
                        # More accurate gradient with averaging over 2 directions
                        g, _, _ = robust_spsa_grad(x, num_dirs=2)
                        evals += 4  # 2 directions * 2 evals each
                        if g is None or np.linalg.norm(g) < 1e-12:
                            break
                        q = g.copy()
                        alpha = np.zeros(len(s_list))
                        for j in range(len(s_list)-1, -1, -1):
                            sy = np.dot(s_list[j], y_list[j])
                            alpha[j] = np.dot(s_list[j], q) / (sy + 1e-30)
                            q = q - alpha[j] * y_list[j]
                        d = -q
                        if len(s_list) > 0:
                            sy_last = np.dot(s_list[-1], y_list[-1])
                            yy_last = np.dot(y_list[-1], y_list[-1])
                            H0 = sy_last / (yy_last + 1e-30)
                            d = H0 * d
                        for j in range(len(s_list)):
                            sy = np.dot(s_list[j], y_list[j])
                            beta = np.dot(y_list[j], d) / (sy + 1e-30)
                            d = d + (alpha[j] - beta) * s_list[j]

                        # Line search with parabolic interpolation
                        alpha_step = 1.0
                        c_armijo = 1e-4
                        f0 = f
                        x_new = None
                        f_new = None
                        # try alpha=1, if fails, do quadratic fit
                        x_try = np.clip(x + alpha_step * d, lb, ub)
                        f_try = func(x_try)
                        evals += 1
                        if f_try <= f0 + c_armijo * alpha_step * np.dot(g, x_try - x):
                            x_new, f_new = x_try, f_try
                        else:
                            # parabolic interpolation with two points: (0,f0) and (alpha_step,f_try)
                            # derivative at 0: np.dot(g, d)
                            g_d = np.dot(g, d)
                            if g_d < -1e-12:
                                # fit parabola f(α) ≈ f0 + g_d*α + (f_try - f0 - g_d*α)/α^2 * α^2
                                # Find minimum: α* = -g_d/(2*(f_try - f0 - g_d*α)/α^2)
                                # Simplified: solve derivative = 0
                                a = (f_try - f0 - g_d * alpha_step) / (alpha_step**2)
                                if a > 0:
                                    alpha_opt = -g_d / (2 * a)
                                    alpha_opt = max(0.01, min(alpha_opt, 2.0))
                                    x_try2 = np.clip(x + alpha_opt * d, lb, ub)
                                    f_try2 = func(x_try2)
                                    evals += 1
                                    if f_try2 < f_try and f_try2 <= f0 + c_armijo * alpha_opt * g_d:
                                        x_new, f_new = x_try2, f_try2
                                    else:
                                        # fallback: half step
                                        alpha_step *= 0.5
                                        x_try_half = np.clip(x + alpha_step * d, lb, ub)
                                        f_try_half = func(x_try_half)
                                        evals += 1
                                        if f_try_half <= f0 + c_armijo * alpha_step * g_d:
                                            x_new, f_new = x_try_half, f_try_half
                            if x_new is None:
                                # try smaller step
                                for _ in range(3):
                                    alpha_step *= 0.5
                                    x_try = np.clip(x + alpha_step * d, lb, ub)
                                    f_try = func(x_try)
                                    evals += 1
                                    if f_try <= f0 + c_armijo * alpha_step * g_d:
                                        x_new, f_new = x_try, f_try
                                        break

                        if x_new is None or alpha_step < 1e-12:
                            break

                        # Compute new gradient for L-BFGS update
                        g_new, _, _ = robust_spsa_grad(x_new, num_dirs=1)
                        evals += 2
                        if g_new is None:
                            break
                        s = x_new - x
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

                # --- If L-BFGS didn't improve enough, try Nelder-Mead ---
                if f >= self.f_opt - 1e-8:
                    # only do NM if we have at least 5+dim budget left
                    if evals + dim + 5 < self.budget:
                        x_nm, f_nm, nm_evals = nelder_mead_step(self.x_opt, self.f_opt, func)
                        evals += nm_evals
                        if f_nm < self.f_opt:
                            self.f_opt = f_nm
                            self.x_opt = x_nm.copy()
                            x = x_nm
                            f = f_nm

                # Inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    if evals < self.budget:
                        perturbed = self.x_opt + 0.01 * np.random.randn(dim) * domain_range
                        perturbed = np.clip(perturbed, lb, ub)
                        f_pert = func(perturbed)
                        evals += 1
                        if f_pert < fitness.max():
                            worst2 = np.argmax(fitness)
                            pop[worst2] = perturbed
                            fitness[worst2] = f_pert
                            if f_pert < self.f_opt:
                                self.f_opt = f_pert
                                self.x_opt = perturbed.copy()

                # Adapt local search frequency
                if f < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                # Orthogonal Latin Hypercube for restart
                # create a Latin hypercube rotated by a random orthogonal matrix
                perm = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm[:, j] = np.random.permutation(perm[:, j])
                lhs = (perm - 0.5) / n_restart
                # random rotation to break linear dependence
                Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                lhs_rot = lhs @ Q.T
                lhs_rot = (lhs_rot - lhs_rot.min(axis=0)) / (lhs_rot.max(axis=0) - lhs_rot.min(axis=0) + 1e-30)
                lhs_rot = lb + lhs_rot * domain_range
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lhs_rot[idx]
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset SHADE memory, archive, L-BFGS memory
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt