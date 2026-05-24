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
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        domain_range = ub - lb

        # --- parameters ---
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)

        # SHADE memory
        mem_size = 10
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_F_success = np.zeros(mem_size)
        mem_CR_success = np.zeros(mem_size)
        mem_weights = np.zeros(mem_size)
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

        archive = []
        archive_size = pop_size * 2

        # stagnation / ls control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(8, int(0.05 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))

        # L‑BFGS memory
        L_mem = 10
        s_list = []
        y_list = []

        success_rates = []

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction (quadratic, slower initial decay)
            ratio = max(0, 1 - (gen / max_gen) ** 0.7)
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

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # r1,r2 from union pop+archive
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

                # Sample F, CR from memory with bias to successful slots
                if np.random.rand() < 0.9 and np.sum(mem_weights) > 0:
                    probs = mem_weights / np.sum(mem_weights)
                    r = np.random.choice(mem_size, p=probs)
                else:
                    r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial 70%, exponential 30%
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
                    # archive
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

            # Update SHADE memory with weighted Lehmer mean
            if len(success_F) > 0:
                w = np.array(weight)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                    CR_mean = np.sum(w * np.array(success_CR))
                    mem_F[mem_idx] = F_lehmer
                    mem_CR[mem_idx] = CR_mean
                    # weight proportional to total improvement in this generation
                    mem_weights[mem_idx] = np.sum(weight)  # use raw sum as weight for future sampling
                    mem_idx = (mem_idx + 1) % mem_size

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Local search (SPSA‑based L‑BFGS) ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 50 and diversity and low_success):

                # Multi‑point SPSA gradient (average of 2 directions)
                c = 1e-3 * domain_range.mean()
                def spsa_grad_avg(x):
                    g_avg = np.zeros(dim)
                    n_pert = 2  # number of random directions
                    evals_local = 0
                    for _ in range(n_pert):
                        delta = np.random.choice([-1, 1], size=dim)
                        x_plus = np.clip(x + c * delta, lb, ub)
                        x_minus = np.clip(x - c * delta, lb, ub)
                        f_plus = func(x_plus)
                        f_minus = func(x_minus)
                        evals_local += 2
                        if f_plus == np.inf or f_minus == np.inf:
                            continue
                        g_avg += (f_plus - f_minus) / (2 * c) * delta
                    return g_avg / n_pert, evals_local

                x = self.x_opt.copy()
                f = self.f_opt
                ls_iters = max(2, min(8, int(0.02 * budget_left / dim)))

                for it in range(ls_iters):
                    if evals + 6 >= self.budget:
                        break
                    g, evals_grad = spsa_grad_avg(x)
                    evals += evals_grad
                    if np.linalg.norm(g) < 1e-12:
                        break

                    # L‑BFGS two‑loop recursion
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        sy = np.dot(s_list[i], y_list[i])
                        if abs(sy) > 1e-30:
                            alpha[i] = np.dot(s_list[i], q) / sy
                        else:
                            alpha[i] = 0
                        q = q - alpha[i] * y_list[i]
                    d = -q
                    if len(s_list) > 0:
                        sy_last = np.dot(s_list[-1], y_list[-1])
                        yy_last = np.dot(y_list[-1], y_list[-1])
                        H0 = sy_last / (yy_last + 1e-30)
                        d = H0 * d
                    for i in range(len(s_list)):
                        sy = np.dot(s_list[i], y_list[i])
                        if abs(sy) > 1e-30:
                            beta = np.dot(y_list[i], d) / sy
                        else:
                            beta = 0
                        d = d + (alpha[i] - beta) * s_list[i]

                    # Quadratic interpolation line search (limited evals)
                    f0 = f
                    x0 = x.copy()
                    alpha_step = 1.0
                    x1 = np.clip(x + alpha_step * d, lb, ub)
                    f1 = func(x1)
                    evals += 1
                    if f1 < f0 + 1e-4 * alpha_step * np.dot(g, x1 - x0):
                        # accept step
                        x_new = x1
                        f_new = f1
                    else:
                        # try a smaller step and fit parabola
                        alpha_step2 = 0.5 * alpha_step
                        x2 = np.clip(x + alpha_step2 * d, lb, ub)
                        f2 = func(x2)
                        evals += 1
                        # Quadratic fit using f0, f2, f1 (with alpha=0, a2, a1)
                        a1 = alpha_step
                        a2 = alpha_step2
                        denom = (a1 - a2) * a1 * a2
                        if abs(denom) > 1e-30:
                            a_opt = 0.5 * ((a2**2 - a1**2) * f0 + (a1**2) * f2 - (a2**2) * f1) / denom
                        else:
                            a_opt = alpha_step2
                        a_opt = max(min(a_opt, 1.0), 0.0)
                        x_opt = np.clip(x + a_opt * d, lb, ub)
                        f_opt = func(x_opt)
                        evals += 1
                        # pick best among visited
                        candidates = [(x1, f1), (x2, f2), (x_opt, f_opt)]
                        best_cand = min(candidates, key=lambda t: t[1])
                        x_new = best_cand[0]
                        f_new = best_cand[1]

                    # Compute new gradient for L‑BFGS update
                    g_new, evals_grad2 = spsa_grad_avg(x_new)
                    evals += evals_grad2
                    if np.linalg.norm(g_new) < 1e-12:
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
                        perturb = self.x_opt + 0.01 * np.random.randn(dim) * domain_range
                        perturb = np.clip(perturb, lb, ub)
                        f_pert = func(perturb)
                        evals += 1
                        if f_pert < fitness.max():
                            worst2 = np.argmax(fitness)
                            pop[worst2] = perturb
                            fitness[worst2] = f_pert
                            if f_pert < self.f_opt:
                                self.f_opt = f_pert
                                self.x_opt = perturb.copy()

                # Adapt local search frequency
                if f_new < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.15))

            # ---------- Stagnation detection and restart ----------
            improvement = best_old - self.f_opt
            if improvement > 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen)):
                n_restart = max(1, int(0.5 * pop_size))
                # Generate new points: half near best, half LHS
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.2 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + lhs[idx] * domain_range
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                mem_weights[:] = 0
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt