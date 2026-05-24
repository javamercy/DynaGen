import numpy as np

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
        D = dim

        # Population size (L-SHADE style)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory for F and CR
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin hypercube initialization (space-filling)
        n_lhs = pop_size
        lhs = np.zeros((n_lhs, dim))
        for j in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs[:, j] = (perm + np.random.uniform(size=n_lhs)) / n_lhs
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
        best_progress_window = [self.f_opt] * 10

        # Local search parameters
        ls_freq_init = max(10, int(0.06 * max_gen))
        ls_freq = ls_freq_init
        # SPSA perturbation factor (adaptive)
        c_base = 1e-3
        # L‑BFGS memory (two‑loop recursion)
        L_mem = 10
        s_list = []
        y_list = []

        # Success history for LS trigger
        success_rates = []
        diversity_history = []

        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / (1.5 * max_gen)))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Adaptive pbest: small early, larger later, also depending on success rate
            avg_success = np.mean(success_rates[-10:]) if len(success_rates) >= 10 else 0.5
            p = 0.1 + 0.3 * (gen / max_gen) - 0.2 * avg_success
            p = np.clip(p, 0.05, 0.5)

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

                # Select r1, r2 from union of pop and archive
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

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

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
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # Success rate
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # Diversity (mean pairwise distance)
            if pop_size > 1:
                centroids = np.mean(pop, axis=0)
                diversity = np.mean(np.sqrt(np.sum((pop - centroids)**2, axis=1)))
            else:
                diversity = 0
            diversity_history.append(diversity)
            if len(diversity_history) > 10:
                diversity_history.pop(0)
            mean_diversity = np.mean(diversity_history) if diversity_history else 0.5

            # ---------- Adaptive local search (L-BFGS with SPSA gradient) ----------
            budget_left = self.budget - evals
            # Trigger conditions: generation frequency, budget, low diversity, and low success rate
            low_success = (len(success_rates) >= 5 and np.mean(success_rates[-5:]) < 0.15)
            low_diversity = (mean_diversity < 0.05 * (ub - lb).mean())
            if (gen % ls_freq == 0 and budget_left > 30 and
                (low_success or low_diversity) and np.std(fitness) < 1.0):

                # Adaptive SPSA perturbation
                c = c_base * (1 + 0.5 * np.log(1 + gen)) * (ub - lb).mean()
                def spsa_grad(x):
                    delta = np.random.choice([-1, 1], size=dim)
                    x_plus = np.clip(x + c * delta, lb, ub)
                    x_minus = np.clip(x - c * delta, lb, ub)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    g = (f_plus - f_minus) / (2 * c) * (1.0 / delta)
                    return g, f_plus, f_minus

                x = self.x_opt.copy()
                f = self.f_opt
                # L-BFGS two-loop recursion with SPSA gradient
                ls_iters = max(2, min(15, int(0.05 * budget_left / dim)))
                for it in range(ls_iters):
                    if evals + 2 >= self.budget:
                        break
                    g, f_plus, f_minus = spsa_grad(x)
                    evals += 2
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break
                    # Compute search direction via L-BFGS
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
                    # Line search (Armijo with quadratic interpolation)
                    alpha_step = 1.0
                    c_armijo = 1e-4
                    f0 = f
                    # Try step 1
                    x_new = np.clip(x + alpha_step * d, lb, ub)
                    if evals >= self.budget:
                        break
                    f_new = func(x_new)
                    evals += 1
                    if f_new <= f0 + c_armijo * alpha_step * np.dot(g, x_new - x):
                        pass  # accepted
                    else:
                        # Backtrack with quadratic interpolation
                        a = alpha_step
                        fa = f_new
                        # use f0 and derivative at x: df0 = np.dot(g, d)
                        df0 = np.dot(g, d)
                        # quadratic model: phi(alpha) = f0 + df0*alpha + (fa - f0 - df0*a) * (alpha/a)^2
                        # find alpha_min that minimizes phi in (0, a]
                        # alpha_min = -df0 * a^2 / (2*(fa - f0 - df0*a))
                        denom = 2 * (fa - f0 - df0 * a)
                        if denom < 1e-12:
                            alpha_step = a * 0.5
                        else:
                            alpha_star = -df0 * a**2 / denom
                            alpha_star = np.clip(alpha_star, 0.1*a, a)
                            alpha_step = alpha_star
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        if evals >= self.budget:
                            break
                        f_new = func(x_new)
                        evals += 1
                        if f_new > f0 + c_armijo * alpha_step * np.dot(g, x_new - x):
                            # further halving until acceptance
                            for _ in range(5):
                                alpha_step *= 0.5
                                x_new = np.clip(x + alpha_step * d, lb, ub)
                                if evals >= self.budget:
                                    break
                                f_new = func(x_new)
                                evals += 1
                                if f_new <= f0 + c_armijo * alpha_step * np.dot(g, x_new - x):
                                    break
                    if alpha_step < 1e-12:
                        break
                    # Update L-BFGS memory
                    s = x_new - x
                    g_new, _, _ = spsa_grad(x_new)
                    evals += 2
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
                # Inject best local point and a perturbed copy into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    if evals < self.budget:
                        perturbed = self.x_opt + 0.02 * np.random.randn(dim) * (ub - lb)
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
            best_progress_window.append(self.f_opt)
            if len(best_progress_window) > 10:
                best_progress_window.pop(0)
            best_rel_improvement = (best_progress_window[0] - self.f_opt) / (abs(self.f_opt) + 1e-30) if best_progress_window[0] != self.f_opt else 0
            stagnation_threshold = max(10, int(0.08 * max_gen))
            if best_rel_improvement < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter > stagnation_threshold:
                n_restart = max(1, int(0.5 * pop_size))
                # Local search on best before restart to ensure local optimum
                if budget_left > 20:
                    # quick Nelder-Mead-like simplex from best point (2*D steps)
                    x_best = self.x_opt.copy()
                    f_best = self.f_opt
                    simplex = [x_best.copy()]
                    for j in range(dim):
                        step = 0.05 * (ub[j] - lb[j])
                        p = x_best.copy()
                        p[j] = np.clip(p[j] + step, lb[j], ub[j])
                        simplex.append(p)
                    # one iteration of Nelder-Mead (reflect worst)
                    # Actually just do a few random perturbations to keep it simple
                    for _ in range(min(5, budget_left // 2)):
                        if evals >= self.budget:
                            break
                        trial = self.x_opt + 0.1 * np.random.randn(dim) * (ub - lb)
                        trial = np.clip(trial, lb, ub)
                        ft = func(trial)
                        evals += 1
                        if ft < f_best:
                            f_best = ft
                            x_best = trial
                            if ft < self.f_opt:
                                self.f_opt = ft
                                self.x_opt = trial.copy()
                # Generate new points: half around current best, half random
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.15 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
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
                ls_freq = min(max_gen // 4, ls_freq + 2)
                best_progress_window = [self.f_opt] * 10

        return self.f_opt, self.x_opt