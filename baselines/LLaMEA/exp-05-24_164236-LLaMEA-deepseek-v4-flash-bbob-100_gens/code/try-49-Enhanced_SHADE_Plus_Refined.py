import numpy as np

class Enhanced_SHADE_Plus_Refined:
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

        # Latin hypercube initial population
        pop = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, j] = lb[j] + (perm + 0.5) / pop_size * (ub[j] - lb[j])
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

        # Success history for adaptive local search trigger
        ls_success_rates = []
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters
        ls_freq_init = max(10, int(0.06 * max_gen))
        ls_freq = ls_freq_init
        L_mem = 7
        s_list = []
        y_list = []

        while evals < self.budget:
            gen += 1

            # Quadratic population size reduction
            ratio = gen / (1.5 * max_gen)
            new_pop_size = max(N_min, int(N_init - (N_init - N_min) * ratio**2))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Adaptive pbest rate based on fitness spread
            spread = np.std(fitness) if pop_size > 1 else 1.0
            p = 0.2 * min(1.0, spread / (ub - lb).mean()) + 0.1
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

                r = np.random.randint(mem_size)
                F = np.clip(mem_F[r] + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(mem_CR[r] + 0.1 * np.random.randn(), 0.0, 1.0)

                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

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
                    weight.append(max(fitness[i] - f_trial, 1e-12))
                    n_success += 1

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # Local search trigger: success rate and budget remaining
            success_rate = n_success / max(1, pop_size)
            ls_success_rates.append(success_rate)
            if len(ls_success_rates) > 10:
                ls_success_rates.pop(0)
            avg_succ = np.mean(ls_success_rates) if ls_success_rates else 0.5

            budget_left = self.budget - evals
            if (gen % ls_freq == 0 and budget_left > 30 and
                np.std(fitness) < 0.5 and avg_succ < 0.2):

                max_ls_iters = max(2, min(10, int(0.03 * budget_left / (dim + 1))))
                c = 1e-3 * (ub - lb).mean()

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
                for it in range(max_ls_iters):
                    if evals + 4 >= self.budget:
                        break
                    g, _, _ = spsa_grad(x)
                    evals += 2
                    if np.linalg.norm(g) < 1e-12:
                        break
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        sy = np.dot(s_list[i], y_list[i])
                        alpha[i] = np.dot(s_list[i], q) / (sy + 1e-30)
                        q = q - alpha[i] * y_list[i]
                    d = -q
                    if len(s_list) > 0:
                        sy_last = np.dot(s_list[-1], y_list[-1])
                        yy_last = np.dot(y_list[-1], y_list[-1])
                        H0 = sy_last / (yy_last + 1e-30)
                        d = H0 * d
                    for i in range(len(s_list)):
                        sy = np.dot(s_list[i], y_list[i])
                        beta = np.dot(y_list[i], d) / (sy + 1e-30)
                        d = d + (alpha[i] - beta) * s_list[i]

                    # Quadratic interpolation line search (reduce evals)
                    alpha_step = 1.0
                    f0 = f
                    x0 = x.copy()
                    for _ in range(5):
                        x_new = np.clip(x0 + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= f0 + 1e-4 * alpha_step * np.dot(g, x_new - x0):
                            break
                        alpha_step *= 0.5
                    else:
                        # If no decrease, try cubic fit using additional point
                        if evals < self.budget:
                            x_mid = np.clip(x0 + 0.5 * alpha_step * d, lb, ub)
                            f_mid = func(x_mid)
                            evals += 1
                            # Use three points for quadratic
                            a0 = f0
                            a1 = (f_mid - f0) / (0.5 * alpha_step)
                            a2 = (f_new - a0 - a1 * alpha_step) / (alpha_step**2 - alpha_step*0.5*alpha_step)
                            if a2 > 0:
                                alpha_step = max(0.1, -a1 / (2*a2))
                                x_new = np.clip(x0 + alpha_step * d, lb, ub)
                                f_new = func(x_new)
                                evals += 1
                            else:
                                alpha_step = 0.1
                                x_new = np.clip(x0 + alpha_step * d, lb, ub)
                                f_new = func(x_new)
                                evals += 1
                        else:
                            break

                    if alpha_step < 1e-12:
                        break
                    s = x_new - x
                    g_new, _, _ = spsa_grad(x_new)
                    evals += 2
                    y = g_new - g
                    if np.dot(s, y) > 1e-10:
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

                # Inject local optimum into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    if evals < self.budget:
                        perturbed = self.x_opt + 0.01 * np.random.randn(dim) * (ub - lb)
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

            # Stagnation detection and restart
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.12 * max_gen)):
                n_restart = max(1, int(0.4 * pop_size))
                # Latin hypercube for restarts
                restart_pop = np.zeros((n_restart, dim))
                for j in range(dim):
                    perm = np.random.permutation(n_restart)
                    restart_pop[:, j] = lb[j] + (perm + 0.5) / n_restart * (ub[j] - lb[j])
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        pop[idx] = restart_pop[idx]
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
                ls_freq = min(max_gen // 4, ls_freq + 2)

        return self.f_opt, self.x_opt