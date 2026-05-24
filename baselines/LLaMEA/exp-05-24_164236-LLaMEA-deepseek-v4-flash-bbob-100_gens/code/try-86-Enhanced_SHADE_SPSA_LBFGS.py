import numpy as np

class Enhanced_SHADE_SPSA_LBFGS:
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

        # Population sizing
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)

        # SHADE memory
        mem_size = 8
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.9)
        mem_idx = 0

        # Latin hypercube initialization
        n_sob = pop_size
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

        # Archive (larger for better diversity)
        archive = []
        archive_size = min(2 * pop_size, int(0.3 * self.budget))

        # Stagnation & local search control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(8, int(0.05 * max_gen))
        min_freq = 4
        max_freq = max(40, int(0.3 * max_gen))

        # L-BFGS memory
        L_mem = 7
        s_list = []
        y_list = []

        # Success history
        success_rates = []
        ls_success = 0
        ls_attempts = 0

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction (faster later)
            ratio = max(0, 1 - (gen / max_gen) ** 1.5)
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate grows
            p = 0.1 + 0.4 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # r1, r2 from union of pop and archive
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

                # Sample F, CR from memory with adaptive noise
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1 (occasional current-to-rand for diversity)
                if np.random.rand() < 0.15:
                    # current-to-rand/1 (exploration)
                    mutant = pop[i] + F * (np.random.rand(dim) - 0.5) * domain_range.mean()
                else:
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover (binomial with higher probability)
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Archive replacement
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
            if len(success_rates) > 15:
                success_rates.pop(0)

            # ---------- Adaptive local search (SPSA-based L-BFGS) ----------
            budget_left = self.budget - evals
            # Trigger: generation frequency, low success rate, low diversity, or stagnation
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.2)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            stagnation_flag = (stagnation_counter > max(3, int(0.02 * max_gen)))
            if (gen % ls_freq == 0 or stagnation_flag) and budget_left > 30 and (diversity or low_success):
                ls_attempts += 1

                # SPSA gradient (with decaying perturbation)
                c = 1e-3 * domain_range.mean() * (1 - evals / self.budget) + 1e-4
                def spsa_grad(x):
                    delta = np.random.choice([-1, 1], size=dim)
                    x_plus = np.clip(x + c * delta, lb, ub)
                    x_minus = np.clip(x - c * delta, lb, ub)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    if f_plus == np.inf or f_minus == np.inf:
                        return None, None, None
                    g = (f_plus - f_minus) / (2 * c) * delta
                    return g, f_plus, f_minus

                # Choose starting point: best or random top-20%
                if np.random.rand() < 0.5 or stagnation_flag:
                    x = self.x_opt.copy()
                    f = self.f_opt
                else:
                    top20 = np.argsort(fitness)[:max(2, int(0.2*pop_size))]
                    idx = np.random.choice(top20)
                    x = pop[idx].copy()
                    f = fitness[idx]

                ls_iters = max(2, min(8, int(0.03 * budget_left / dim)))

                for it in range(ls_iters):
                    if evals + 3 >= self.budget:
                        break
                    g, f_plus, f_minus = spsa_grad(x)
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break
                    evals += 2

                    # L-BFGS two-loop recursion
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        sy = np.dot(s_list[i], y_list[i])
                        if sy == 0:
                            alpha[i] = 0
                        else:
                            alpha[i] = np.dot(s_list[i], q) / sy
                        q = q - alpha[i] * y_list[i]
                    d = -q
                    if len(s_list) > 0:
                        sy_last = np.dot(s_list[-1], y_list[-1])
                        yy_last = np.dot(y_list[-1], y_list[-1])
                        H0 = sy_last / (yy_last + 1e-30)
                        d = H0 * d
                    for i in range(len(s_list)):
                        sy = np.dot(s_list[i], y_list[i])
                        if sy == 0:
                            beta = 0
                        else:
                            beta = np.dot(y_list[i], d) / sy
                        d = d + (alpha[i] - beta) * s_list[i]

                    # Line search with quadratic interpolation
                    alpha_step = 1.0
                    c_armijo = 1e-4
                    f0 = f
                    x_new = None
                    f_new = None
                    f_prev = None
                    alpha_prev = None
                    for _ in range(8):
                        x_try = np.clip(x + alpha_step * d, lb, ub)
                        f_try = func(x_try)
                        evals += 1
                        if f_try <= f0 + c_armijo * alpha_step * np.dot(g, x_try - x):
                            x_new = x_try
                            f_new = f_try
                            break
                        # Quadratic interpolation for better step
                        if f_prev is not None and alpha_prev is not None:
                            denom = 2 * (alpha_step * f_prev - alpha_prev * f_try + (alpha_prev - alpha_step) * f0)
                            if denom != 0:
                                alpha_quad = (alpha_step**2 * f_prev - alpha_prev**2 * f_try) / denom
                                alpha_quad = max(0.1 * alpha_step, min(0.9 * alpha_step, alpha_quad))
                                alpha_step = alpha_quad
                                f_prev = None
                                alpha_prev = None
                                continue
                        f_prev = f_try
                        alpha_prev = alpha_step
                        alpha_step *= 0.5
                    if x_new is None or alpha_step < 1e-12:
                        break

                    # Compute new gradient for L-BFGS update
                    g_new, _, _ = spsa_grad(x_new)
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

                # Inject best found into population
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

                # Adapt LS frequency based on success
                if f_new is not None and f_new < self.f_opt - 1e-8:
                    ls_success += 1
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.15))
                ls_freq = np.clip(ls_freq, min_freq, max_freq)

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.1 * max_gen)):
                # Restart: keep best 20%, replace worst 40% with random perturbations, rest with LHS
                elite_n = max(1, int(0.2 * pop_size))
                rest1 = int(0.4 * pop_size)
                rest2 = pop_size - elite_n - rest1
                idx_sorted = np.argsort(fitness)
                new_pop = []
                new_fit = []
                # Keep elite
                for idx in idx_sorted[:elite_n]:
                    new_pop.append(pop[idx].copy())
                    new_fit.append(fitness[idx])
                # Random perturbations around best
                for _ in range(rest1):
                    if len(new_pop) > 0:
                        base = new_pop[np.random.randint(len(new_pop))]
                        new_ind = base + 0.1 * np.random.randn(dim) * domain_range
                    else:
                        new_ind = self.x_opt + 0.1 * np.random.randn(dim) * domain_range
                    new_ind = np.clip(new_ind, lb, ub)
                    f_val = func(new_ind)
                    evals += 1
                    new_pop.append(new_ind)
                    new_fit.append(f_val)
                    if f_val < self.f_opt:
                        self.f_opt = f_val
                        self.x_opt = new_ind.copy()
                # Latin hypercube for the rest
                perm2 = np.tile(np.arange(1, rest2 + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / rest2
                for k in range(rest2):
                    new_ind = lb + lhs[k] * domain_range
                    f_val = func(new_ind)
                    evals += 1
                    new_pop.append(new_ind)
                    new_fit.append(f_val)
                    if f_val < self.f_opt:
                        self.f_opt = f_val
                        self.x_opt = new_ind.copy()
                pop = np.array(new_pop)
                fitness = np.array(new_fit)
                # Reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.9
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt