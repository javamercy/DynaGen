import numpy as np

class Enhanced_SHADE_AdamLS:
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

        # population size parameters
        N_init = max(10, int(16 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.0)

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # quasi-random initialization (Sobol-like Latin hypercube)
        n_init = pop_size
        perm = np.tile(np.arange(1, n_init + 1), (dim, 1)).T
        for j in range(dim):
            perm[:, j] = np.random.permutation(perm[:, j])
        sobol = (perm - 0.5) / n_init
        pop = lb + sobol * domain_range

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        archive = []
        archive_size = pop_size

        # Adam local search parameters
        beta1 = 0.9
        beta2 = 0.999
        eps_adam = 1e-8
        m_adam = np.zeros(dim)
        v_adam = np.zeros(dim)
        t_adam = 0

        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(6, int(0.04 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))

        # success history for LS trigger
        success_rates = []

        # covariance estimate for restart
        cov = np.eye(dim) * (0.2 * domain_range.mean())**2

        while evals < self.budget:
            gen += 1

            # nonlinear population reduction (slower than original)
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

            p = 0.1 + 0.4 * (gen / max_gen) ** 1.0
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

                # sample F, CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # mutation current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # crossover (binomial with probability 0.7, exponential 0.3)
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

            # update SHADE memory
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

            # success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Adam local search with SPSA gradients ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.2)
            diversity = np.std(fitness) < 0.4 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 20 and diversity and low_success):
                c = 1e-3 * domain_range.mean()
                x = self.x_opt.copy()
                f = self.f_opt
                # reset Adam state for this local search
                m_adam[:] = 0.0
                v_adam[:] = 0.0
                t_adam = 0
                ls_iters = min(8, max(2, int(0.03 * budget_left / dim)))
                for it in range(ls_iters):
                    if evals + 3 >= self.budget:
                        break
                    # SPSA gradient
                    delta = np.random.choice([-1, 1], size=dim)
                    x_plus = np.clip(x + c * delta, lb, ub)
                    x_minus = np.clip(x - c * delta, lb, ub)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    evals += 2
                    if f_plus == np.inf or f_minus == np.inf:
                        break
                    g = (f_plus - f_minus) / (2 * c) * delta
                    if np.linalg.norm(g) < 1e-12:
                        break
                    # Adam update
                    t_adam += 1
                    m_adam = beta1 * m_adam + (1 - beta1) * g
                    v_adam = beta2 * v_adam + (1 - beta2) * (g ** 2)
                    m_hat = m_adam / (1 - beta1 ** t_adam)
                    v_hat = v_adam / (1 - beta2 ** t_adam)
                    step = 0.5 * m_hat / (np.sqrt(v_hat) + eps_adam)  # 0.5 as base step
                    # backtracking line search (up to 4 evaluations)
                    alpha = 0.5  # base step size scaled by domain range
                    step_norm = np.linalg.norm(step)
                    if step_norm > 0:
                        step = step / step_norm * np.minimum(0.1 * domain_range.mean(), 1.0)
                    f0 = f
                    x_new = None
                    f_new = None
                    step_scales = [1.0, 0.5, 0.25, 0.125]
                    for scale in step_scales:
                        x_try = np.clip(x + scale * step, lb, ub)
                        f_try = func(x_try)
                        evals += 1
                        if f_try < f0:
                            x_new = x_try
                            f_new = f_try
                            break
                    if x_new is None:
                        break
                    x = x_new
                    f = f_new
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()

                # inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # also add a perturbed copy
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

                # adapt LS frequency
                if f_new is not None and f_new < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.15))

            # ---------- Stagnation detection and covariance-aware restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
                # update covariance estimate from recent steps (if any)
                if evals > 0:
                    cov = 0.9 * cov + 0.1 * np.outer(self.x_opt - np.mean(pop, axis=0), self.x_opt - np.mean(pop, axis=0))
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.1 * max_gen)):
                n_restart = max(2, int(0.5 * pop_size))
                # generate restarts: half from Gaussian around best with adaptive cov, half from Latin hypercube
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # sample from multivariate Gaussian with learned covariance (scaled)
                        try:
                            L = np.linalg.cholesky(cov + 1e-6 * np.eye(dim))
                            z = np.random.randn(dim)
                            pop[idx] = self.x_opt + L @ z * 0.5
                        except:
                            pop[idx] = self.x_opt + 0.1 * np.random.randn(dim) * domain_range
                    else:
                        pop[idx] = lb + lhs[idx] * domain_range
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # keep SHADE memories but reset archive and stagnation counter
                archive.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt