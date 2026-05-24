import numpy as np

class Adaptive_SHADE_DualMut_LS:
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
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Dual mutation memory
        mut_success = [0, 0]
        mut_attempts = [0, 0]
        p_mut = 0.7

        # Initial population (Latin hypercube)
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
        archive_size = pop_size

        # Local search control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(8, int(0.05 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))
        ls_freq_current = ls_freq

        # L-BFGS memory
        L_mem = 5
        s_list = []
        y_list = []
        ls_step = 1.0

        # Success history
        success_rates = []

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction
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

            # pbest rate
            p = 0.1 + 0.4 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                use_pbest = np.random.rand() < p_mut

                # Select two individuals from union of pop and archive
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

                # Mutation
                if use_pbest:
                    pbest_size = max(2, int(p * pop_size))
                    best_indices = np.argsort(fitness)[:pbest_size]
                    pbest_idx = np.random.choice(best_indices)
                    x_pbest = pop[pbest_idx]
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                else:
                    mutant = x_r1 + F * (x_r2 - x_r1)  # DE/rand/1

                # Crossover (binomial or exponential)
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

                # Update mutation strategy counters
                if use_pbest:
                    mut_attempts[0] += 1
                    if f_trial <= fitness[i]:
                        mut_success[0] += 1
                else:
                    mut_attempts[1] += 1
                    if f_trial <= fitness[i]:
                        mut_success[1] += 1

                if f_trial <= fitness[i]:
                    # Archive update
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

            # Update mutation selection probability
            if mut_attempts[0] + mut_attempts[1] > 0:
                rate0 = mut_success[0] / max(1, mut_attempts[0])
                rate1 = mut_success[1] / max(1, mut_attempts[1])
                alpha = 0.2
                p_mut = p_mut * (1 - alpha) + alpha * (rate0 / (rate0 + rate1 + 1e-30))
                p_mut = np.clip(p_mut, 0.1, 0.9)

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # Adaptive local search (SPSA-based L-BFGS)
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            if (gen % ls_freq_current == 0 and budget_left > 30 and diversity and low_success):

                # SPSA gradient with adaptive perturbation
                c = 1e-3 * domain_range.mean() * (1 + 0.5 * (gen / max_gen))
                def spsa_grad(x):
                    delta = np.random.choice([-1, 1], size=dim)
                    x_plus = np.clip(x + c * delta, lb, ub)
                    x_minus = np.clip(x - c * delta, lb, ub)
                    f_plus = func(x_plus)
                    evals_local = 1
                    f_minus = func(x_minus)
                    evals_local += 1
                    if f_plus == np.inf or f_minus == np.inf:
                        return None, None, None, evals_local
                    g = (f_plus - f_minus) / (2 * c) * delta
                    return g, f_plus, f_minus, evals_local

                x = self.x_opt.copy()
                f = self.f_opt
                ls_iters = max(2, min(10, int(0.03 * budget_left / dim)))

                for it in range(ls_iters):
                    if evals + 4 >= self.budget:
                        break
                    g, f_plus, f_minus, evals_grad = spsa_grad(x)
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break
                    evals += evals_grad

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

                    # Line search with parabolic interpolation
                    alpha_step = ls_step
                    c_armijo = 1e-4
                    f0 = f
                    x_new = None
                    f_new = None

                    # Try initial step
                    x_try = np.clip(x + alpha_step * d, lb, ub)
                    f_try = func(x_try)
                    evals += 1
                    if f_try <= f0 + c_armijo * alpha_step * np.dot(g, x_try - x):
                        x_new = x_try
                        f_new = f_try
                    else:
                        # Try half step
                        alpha_step2 = alpha_step * 0.5
                        x_try2 = np.clip(x + alpha_step2 * d, lb, ub)
                        f_try2 = func(x_try2)
                        evals += 1
                        if f_try2 <= f0 + c_armijo * alpha_step2 * np.dot(g, x_try2 - x):
                            x_new = x_try2
                            f_new = f_try2
                        else:
                            # Parabolic interpolation
                            a1 = alpha_step
                            a2 = alpha_step2
                            denom = (a1 - a2)*(a1*f0 - a1*f_try2 + a2*f_try - a2*f0)
                            if abs(denom) > 1e-12:
                                a_opt = 0.5 * (a1*a1*(f0 - f_try2) + a2*a2*(f_try - f0)) / denom
                                a_opt = np.clip(a_opt, 0, 2*alpha_step)
                                x_try3 = np.clip(x + a_opt * d, lb, ub)
                                f_try3 = func(x_try3)
                                evals += 1
                                if f_try3 < min(f_try, f_try2):
                                    x_new = x_try3
                                    f_new = f_try3
                                else:
                                    best_f = min(f0, f_try, f_try2)
                                    if f_try == best_f:
                                        x_new = x_try; f_new = f_try
                                    else:
                                        x_new = x_try2; f_new = f_try2
                            else:
                                if f_try < f_try2:
                                    x_new = x_try; f_new = f_try
                                else:
                                    x_new = x_try2; f_new = f_try2

                    if x_new is None or np.linalg.norm(x_new - x) < 1e-12:
                        break

                    # Compute new gradient for update
                    g_new, _, _, evals_grad2 = spsa_grad(x_new)
                    evals += evals_grad2
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
                    ls_step = np.linalg.norm(s) / max(1e-12, np.linalg.norm(d))
                    ls_step = np.clip(ls_step, 0.01, 10.0)
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()

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
                if f_new is not None and f_new < self.f_opt - 1e-8:
                    ls_freq_current = max(min_freq, int(ls_freq_current * 0.9))
                else:
                    ls_freq_current = min(max_freq, int(ls_freq_current * 1.1))
            else:
                if stagnation_counter > 10:
                    ls_freq_current = min(max_freq, int(ls_freq_current * 1.02))

            # Stagnation detection and restart
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * domain_range * (1 - gen / max_gen) + 0.01
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
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq_current = max(ls_freq_current, min_freq)

        return self.f_opt, self.x_opt