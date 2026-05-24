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

        # Population sizing — larger initial for better exploration
        N_init = max(12, int(15 * np.sqrt(dim)))
        N_min = 6
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)

        # SHADE memory
        mem_size = 10
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin hypercube initialisation
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

        # Archive
        archive = []
        archive_size = pop_size

        # Stagnation and local search control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(6, int(0.03 * max_gen))
        min_freq = 3
        max_freq = max(25, int(0.15 * max_gen))

        # L‑BFGS memory
        L_mem = 10
        s_list = []
        y_list = []

        # Success history
        success_rates = []

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction (cosine schedule)
            ratio = max(0, (np.cos(gen / max_gen * np.pi / 2)) ** 2)
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate: grows sigmoidally
            p = 0.15 + 0.45 / (1 + np.exp(-8 * (gen / max_gen - 0.5)))
            p = min(p, 0.6)

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

                # r1, r2 from union
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

                # Sample F, CR
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial or exponential (30%)
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
                    mem_F[mem_idx] = min(F_lehmer, 0.95)
                    mem_CR[mem_idx] = np.clip(CR_mean, 0.0, 1.0)
                    mem_idx = (mem_idx + 1) % mem_size

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Adaptive local search (SPSA-based L‑BFGS) ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.1)
            diversity = np.std(fitness) < 0.4 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 50 and diversity and low_success):

                # SPSA gradient with smaller perturbation
                c = 1e-4 * domain_range.mean()
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

                x = self.x_opt.copy()
                f = self.f_opt
                ls_iters = max(2, min(12, int(0.04 * budget_left / dim)))

                for it in range(ls_iters):
                    if evals + 3 >= self.budget:
                        break
                    g, _, _ = spsa_grad(x)
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break
                    evals += 2

                    # L‑BFGS two-loop recursion
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

                    # Line search with quadratic interpolation (fewer evals)
                    alpha_step = 1.0
                    c_armijo = 1e-4
                    f0 = f
                    x_new = None
                    f_new = None
                    # Try step 1.0 first
                    x_try = np.clip(x + alpha_step * d, lb, ub)
                    f_try = func(x_try)
                    evals += 1
                    if f_try <= f0 + c_armijo * alpha_step * np.dot(g, x_try - x):
                        x_new = x_try
                        f_new = f_try
                    else:
                        # Quadratic interpolation: evaluate at alpha=0.5
                        alpha_step = 0.5
                        x_try2 = np.clip(x + alpha_step * d, lb, ub)
                        f_try2 = func(x_try2)
                        evals += 1
                        # If sufficient, use it; else try even smaller step
                        if f_try2 <= f0 + c_armijo * alpha_step * np.dot(g, x_try2 - x):
                            x_new = x_try2
                            f_new = f_try2
                        else:
                            # Perform one more quadratic fit using f0, f_try, f_try2
                            a = f0
                            b = (f_try - f0) / 1.0
                            c = (f_try2 - f0 - b * 0.5) / (0.5 ** 2)
                            # Minimizer = -b/(2c)
                            if c > 0:
                                alpha_opt = max(0.1, min(0.9, -b / (2 * c)))
                            else:
                                alpha_opt = 0.25
                            x_try3 = np.clip(x + alpha_opt * d, lb, ub)
                            f_try3 = func(x_try3)
                            evals += 1
                            if f_try3 <= f0 + c_armijo * alpha_opt * np.dot(g, x_try3 - x):
                                x_new = x_try3
                                f_new = f_try3
                            else:
                                break  # no improvement, exit line search
                    if x_new is None:
                        break

                    # Update L‑BFGS
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

                # Inject best and perturbed copies
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    if evals < self.budget:
                        perturbed = self.x_opt + 0.02 * np.random.randn(dim) * domain_range
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
                if f_new is not None and f_new < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.1 * max_gen)):
                n_restart = max(1, int(0.5 * pop_size))
                # Generate new points: 30% near best, 30% Latin hypercube, 40% uniform random
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < int(0.3 * n_restart):
                        scale = 0.1 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    elif idx < int(0.6 * n_restart):
                        pop[idx] = lb + lhs[idx] * domain_range
                    else:
                        pop[idx] = np.random.uniform(lb, ub)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset SHADE memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = min(ls_freq, min_freq + 2)

        return self.f_opt, self.x_opt