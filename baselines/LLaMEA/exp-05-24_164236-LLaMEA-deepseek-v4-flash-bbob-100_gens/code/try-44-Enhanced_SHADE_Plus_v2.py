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
        lb = -5.0 * np.ones(dim)
        ub = 5.0 * np.ones(dim)

        # Population size (L-SHADE style, slightly larger)
        N_init = max(10, int(16 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)  # more generations

        # SHADE memory for F and CR
        mem_size = 8
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube initial population
        pop = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, j] = lb[j] + (perm + np.random.uniform(0,1,pop_size)) / pop_size * (ub[j] - lb[j])

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
        no_improve_threshold = max(10, int(0.15 * max_gen))

        # Local search parameters
        ls_freq_init = max(8, int(0.05 * max_gen))
        ls_freq = ls_freq_init
        success_rates = []
        ls_iters_max = lambda budget_left: min(15, max(3, int(0.05 * budget_left / (dim + 1))))

        # L-BFGS memory
        L_mem = 10
        s_list = []
        y_list = []

        while evals < self.budget:
            gen += 1

            # Adaptive population size reduction (slower)
            phase = gen / max_gen
            new_pop_size = max(N_min, int(N_init - phase**1.5 * (N_init - N_min)))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate: small early, larger later
            p = 0.15 + 0.35 * (gen / max_gen) ** 1.5
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
                F = np.clip(F, 0.1, 0.9)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1 (probability 0.7) or current-to-rand/1 (0.3) for diversity
                if np.random.rand() < 0.7:
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                else:
                    # Use a different r1,r2 pair for the random part
                    indices2 = [j for j in range(pop_size) if j != i]
                    if len(indices2) >= 2:
                        r3, r4 = np.random.choice(indices2, 2, replace=False)
                    else:
                        r3, r4 = 0, 1
                    mutant = pop[i] + F * (pop[r3] - pop[r4]) + F * (x_r1 - x_r2)

                # Crossover: binomial or exponential (50/50)
                trial = np.zeros(dim)
                if np.random.rand() < 0.5:
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

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Adaptive local search (L-BFGS with SPSA gradient) ----------
            budget_left = self.budget - evals
            # Trigger conditions: low success rate, enough budget, population converged
            success_avg = np.mean(success_rates[-5:]) if len(success_rates) >= 5 else 0.5
            if (gen % ls_freq == 0 and
                budget_left > 2 * (dim + 1) and
                np.std(fitness) < 0.3 * (ub - lb).mean() and
                success_avg < 0.2):

                ls_iters = ls_iters_max(budget_left)
                # SPSA gradient estimation with adaptive step size
                c = 1e-3 * (ub - lb).mean() * (1.0 / (1.0 + 0.1 * gen))
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
                    # Line search (Armijo) using function evaluations
                    alpha_step = 1.0
                    c_armijo = 1e-4
                    f0 = f
                    for _ in range(12):
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= f0 + c_armijo * alpha_step * np.dot(g, x_new - x):
                            break
                        alpha_step *= 0.5
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
                # Inject best local point and perturbed copies into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # Inject a few more perturbed points if budget allows
                    for _ in range(min(2, budget_left - evals)):
                        if evals >= self.budget:
                            break
                        perturbed = self.x_opt + 0.02 * np.random.randn(dim) * (ub - lb) * min(1.0, gen/max_gen)
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
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > no_improve_threshold:
                n_restart = max(2, int(0.6 * pop_size))
                # Generate quasi-random points (Latin hypercube on a reduced domain around current best)
                domain_scale = 0.2 * (1.0 - gen / max_gen) + 0.01
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # Perturb around best
                        pop[idx] = self.x_opt + np.random.uniform(-1,1,dim) * (ub - lb) * domain_scale
                    else:
                        # Uniform in whole domain
                        pop[idx] = np.random.uniform(lb, ub, dim)
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
                ls_freq = min(max_gen // 3, ls_freq + 3)

        return self.f_opt, self.x_opt