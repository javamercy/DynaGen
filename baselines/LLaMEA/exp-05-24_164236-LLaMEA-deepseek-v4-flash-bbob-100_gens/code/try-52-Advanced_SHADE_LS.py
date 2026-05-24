import numpy as np
from scipy.linalg import sqrtm

class Advanced_SHADE_LS:
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
        N_init = max(10, int(15 * np.sqrt(dim)))
        N_min = 5
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)

        # SHADE memory
        mem_size = 8
        mem_F = np.full(mem_size, 0.6)
        mem_CR = np.full(mem_size, 0.9)
        mem_idx = 0

        # Covariance matrix for guided mutation (learned from successful steps)
        C = np.eye(dim)
        C_learning_rate = 0.2

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

        archive = []
        archive_size = int(2.5 * pop_size)

        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search control
        ls_freq = max(6, int(0.04 * max_gen))
        min_freq = 3
        max_freq = max(25, int(0.15 * max_gen))

        # L-BFGS memory
        L_mem = 8
        s_list = []
        y_list = []

        # Success rates for dynamic adaptation
        success_rates = []
        # For C update
        step_buffer = []

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction (faster decay)
            ratio = max(0, 1 - (gen / max_gen) ** 1.5)
            new_pop_size = max(N_min, int(N_init * 0.3 + (N_init - N_min) * ratio * 0.7))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # pbest rate: grows with generation
            p = 0.15 + 0.5 * (gen / max_gen) ** 1.3
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

                # Union of pop and archive for r1, r2
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

                # Covariance-guided mutation: current-to-pbest/1 with rotation
                # Use Cholesky of C to transform differences
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                diff1 = L @ (x_pbest - pop[i]) / np.linalg.norm(x_pbest - pop[i] + 1e-12)
                diff2 = L @ (x_r1 - x_r2)
                mutant = pop[i] + F * diff1 + F * diff2

                # Crossover: binomial (75%) or exponential (25%)
                if np.random.rand() < 0.75:
                    j_rand = np.random.randint(dim)
                    mask = np.random.rand(dim) < CR
                    mask[j_rand] = True
                    trial = np.where(mask, mutant, pop[i])
                else:
                    start = np.random.randint(dim)
                    L_len = 0
                    while L_len < dim and np.random.rand() < CR:
                        L_len += 1
                    indices = (np.arange(dim) + start) % dim
                    mask = np.zeros(dim, dtype=bool)
                    mask[indices[:L_len]] = True
                    trial = np.where(mask, mutant, pop[i])

                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Archive replacement
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        archive[np.random.randint(len(archive))] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))
                    n_success += 1

                    # Record step for covariance matrix update
                    step = trial - pop[i]
                    step_buffer.append(step)
                    if len(step_buffer) > dim * 5:
                        step_buffer.pop(0)

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

            # Update covariance matrix C from successful steps
            if len(step_buffer) >= dim:
                steps = np.array(step_buffer)
                cov_new = np.cov(steps, rowvar=False)
                if cov_new.shape == (dim, dim) and not np.any(np.isnan(cov_new)):
                    C = (1 - C_learning_rate) * C + C_learning_rate * cov_new
                    # Regularization
                    C += 1e-10 * np.eye(dim)

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Adaptive local search (SPSA-based L-BFGS) ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.2)
            diversity = np.std(fitness) < 0.3 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 50 and diversity and low_success):

                # Robust gradient estimation: average multiple SPSA samples
                c = 1e-3 * domain_range.mean()
                n_grad_samples = max(3, min(6, int(budget_left / 20)))
                grad_accum = np.zeros(dim)
                f_plus_accum = 0.0
                f_minus_accum = 0.0
                for _ in range(n_grad_samples):
                    delta = np.random.choice([-1, 1], size=dim)
                    x_plus = np.clip(self.x_opt + c * delta, lb, ub)
                    x_minus = np.clip(self.x_opt - c * delta, lb, ub)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    evals += 2
                    if f_plus == np.inf or f_minus == np.inf:
                        continue
                    grad_accum += (f_plus - f_minus) / (2 * c) * delta
                    f_plus_accum += f_plus
                    f_minus_accum += f_minus
                if n_grad_samples > 0:
                    grad_accum /= n_grad_samples
                    g = grad_accum
                else:
                    g = None

                if g is not None and np.linalg.norm(g) > 1e-12:
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

                    # Line search (Armijo)
                    alpha_step = 1.0
                    c_armijo = 1e-4
                    x = self.x_opt.copy()
                    f = self.f_opt
                    ls_iters = max(2, min(8, int(0.02 * budget_left / dim)))
                    for it_ls in range(ls_iters):
                        if evals + 2 >= self.budget:
                            break
                        x_try = np.clip(x + alpha_step * d, lb, ub)
                        f_try = func(x_try)
                        evals += 1
                        if f_try <= f + c_armijo * alpha_step * np.dot(g, x_try - x):
                            # Compute new gradient
                            grad_new_accum = np.zeros(dim)
                            for _ in range(max(2, n_grad_samples // 2)):
                                delta = np.random.choice([-1, 1], size=dim)
                                x_plus = np.clip(x_try + c * delta, lb, ub)
                                x_minus = np.clip(x_try - c * delta, lb, ub)
                                f_plus = func(x_plus)
                                f_minus = func(x_minus)
                                evals += 2
                                if f_plus == np.inf or f_minus == np.inf:
                                    continue
                                grad_new_accum += (f_plus - f_minus) / (2 * c) * delta
                            if np.linalg.norm(grad_new_accum) < 1e-12:
                                break
                            g_new = grad_new_accum / max(1, n_grad_samples // 2)
                            s = x_try - x
                            y = g_new - g
                            sy = np.dot(s, y)
                            if sy > 1e-10:
                                if len(s_list) >= L_mem:
                                    s_list.pop(0)
                                    y_list.pop(0)
                                s_list.append(s.copy())
                                y_list.append(y.copy())
                            x = x_try
                            f = f_try
                            g = g_new
                            if f < self.f_opt:
                                self.f_opt = f
                                self.x_opt = x.copy()
                            break
                        alpha_step *= 0.5
                    # Inject best into population
                    if self.f_opt < fitness.max():
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = self.x_opt.copy()
                        fitness[worst_idx] = self.f_opt
                        # Generate a perturbed copy
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

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-9:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.07 * max_gen)):
                n_restart = max(2, int(0.5 * pop_size))
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 3:
                        scale = 0.2 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    elif idx < 2 * n_restart // 3:
                        pop[idx] = lb + lhs[idx] * domain_range
                    else:
                        # Local restart around best
                        pop[idx] = self.x_opt + np.random.randn(dim) * domain_range * 0.05
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset SHADE memory and archives
                mem_F[:] = 0.6
                mem_CR[:] = 0.9
                archive.clear()
                s_list.clear()
                y_list.clear()
                step_buffer.clear()
                C = np.eye(dim)
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt