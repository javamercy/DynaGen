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
        # robust bounds handling
        if hasattr(func, 'bounds') and hasattr(func.bounds, 'lb'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        elif hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.full(dim, -5.0)
            ub = np.full(dim, 5.0)

        # Population size reduction (L-SHADE style)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling initial population
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

        # Local search parameters
        ls_freq_init = max(10, int(0.05 * max_gen))
        ls_freq = ls_freq_init
        ls_max_iter = max(3, min(6, int(0.02 * (self.budget / (dim + 5)))))
        L_mem = 5
        s_list = []
        y_list = []

        # Success history for adaptation
        success_rates = []
        best_improvements = []

        while evals < self.budget:
            gen += 1

            # Linear population size reduction (slower)
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / (1.5 * max_gen)))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Dynamic pbest rate – decreasing over time
            p = 0.2 * (1.0 - evals / self.budget)**2 + 0.1
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

                # Sample F, CR from memory with noise
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

            # Success rate and best improvement tracking
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 15:
                success_rates.pop(0)
            if len(best_improvements) > 10:
                best_improvements.pop(0)
            best_improvements.append(self.f_opt)

            # Adaptive local search frequency
            avg_success = np.mean(success_rates[-5:]) if len(success_rates) >= 5 else 0.0
            if avg_success < 0.1 and ls_freq > 5:
                ls_freq -= 1
            elif avg_success > 0.3 and ls_freq < 2 * ls_freq_init:
                ls_freq += 1

            # ---------- Adaptive local search (L-BFGS) ----------
            # Trigger conditions: low diversity, low improvement, enough budget, and not too frequent
            cond_diversity = (np.std(fitness) < 0.5) if pop_size > 1 else True
            cond_improvement = (len(best_improvements) >= 10 and
                                abs(best_improvements[-1] - best_improvements[-10]) < 1e-6)
            cond_budget = (self.budget - evals) > dim * 10 + 20
            if (gen % ls_freq == 0 and cond_diversity and cond_budget and
                (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15 or cond_improvement)):

                # Choose gradient method: central differences for low dim, otherwise forward
                use_central = dim < 10
                h = 1e-6 * (ub - lb) + 1e-10
                x = self.x_opt.copy()
                f = self.f_opt

                def grad_fwd(x, fx):
                    g = np.zeros(dim)
                    for d in range(dim):
                        x_plus = np.clip(x + np.eye(1, dim, d) * h[d], lb, ub)[0]
                        g[d] = (func(x_plus) - fx) / h[d]
                    return g

                def grad_central(x):
                    g = np.zeros(dim)
                    for d in range(dim):
                        x_plus = np.clip(x + np.eye(1, dim, d) * h[d], lb, ub)[0]
                        x_minus = np.clip(x - np.eye(1, dim, d) * h[d], lb, ub)[0]
                        g[d] = (func(x_plus) - func(x_minus)) / (2 * h[d])
                    return g

                for it in range(ls_max_iter):
                    if evals + dim + (dim if use_central else 1) >= self.budget:
                        break
                    # Compute gradient
                    if use_central:
                        g = grad_central(x)
                        evals += 2 * dim
                    else:
                        if it == 0:
                            fx = f
                        else:
                            fx = func(x)
                            evals += 1
                        g = grad_fwd(x, fx)
                        evals += dim
                    if np.linalg.norm(g) < 1e-12:
                        break
                    # L-BFGS two-loop recursion
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
                    # Line search (Armijo, limited to 5 evaluations)
                    alpha_step = 1.0
                    c = 1e-4
                    for _ in range(5):
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
                    if use_central:
                        g_new = grad_central(x_new)
                        evals += 2 * dim
                    else:
                        g_new = grad_fwd(x_new, f_new)
                        evals += dim
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
                # Inject best local point and a perturbed copy
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    if evals < self.budget:
                        perturbed = self.x_opt + 0.005 * np.random.randn(dim) * (ub - lb)
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

            # ---------- Stagnation triggered restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.1 * max_gen)):
                n_restart = max(1, int(0.5 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Quasi-random LHS for global points
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
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

        return self.f_opt, self.x_opt