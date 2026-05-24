import numpy as np

class Hyper_Adaptive_SHADE_SPSA_LBFGS:
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

        # Population sizing (L‑SHADE style: start larger, reduce to N_min)
        N_init = max(8, int(16 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.2)

        # SHADE memory
        mem_size = 8
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube initialization (quasi‑random)
        n = pop_size
        perm = np.tile(np.arange(1, n+1), (dim, 1)).T
        for j in range(dim):
            perm[:, j] = np.random.permutation(perm[:, j])
        lhs = (perm - 0.5) / n
        pop = lb + lhs * domain_range

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

        # Stagnation counters
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search control (frequency based on recent success rate)
        base_ls_freq = max(10, int(0.06 * max_gen))
        ls_freq = base_ls_freq
        min_freq = 4
        max_freq = max(40, int(0.2 * max_gen))
        success_rates = []

        # L‑BFGS memory
        L_mem = 6
        s_list = []
        y_list = []

        while evals < self.budget:
            gen += 1

            # ---------- Nonlinear population reduction (power law) ----------
            ratio = max(0, 1 - (gen / max_gen) ** 1.5)
            new_pop_size = max(N_min, int(N_init * ratio + 0.5))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate: increase with generation
            p = 0.1 + 0.5 * (gen / max_gen) ** 1.5
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

                # Sample F, CR from memory with small perturbations
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.05 * np.random.randn()
                CR = mem_CR[r] + 0.05 * np.random.randn()
                F = np.clip(F, 0.2, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial (80%) or exponential (20%)
                if np.random.rand() < 0.8:
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
                    # Archive replacement (FIFO style)
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        archive[gen % archive_size] = pop[i].copy()

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

            # Update SHADE memory (weighted Lehmer mean)
            if len(success_F) > 0:
                w = np.array(weight)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                    CR_mean = np.sum(w * np.array(success_CR))
                    mem_F[mem_idx] = F_lehmer
                    mem_CR[mem_idx] = np.clip(CR_mean, 0., 1.)
                    mem_idx = (mem_idx + 1) % mem_size

            # Success rate for local search adaptation
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)
            recent_success = np.mean(success_rates[-5:]) if len(success_rates) >= 5 else 1.0

            # ---------- SPSA‑based L‑BFGS local search ----------
            budget_left = self.budget - evals
            diversity = np.std(fitness) < 0.4 * np.mean(domain_range)
            trigger = (gen % ls_freq == 0) and (budget_left > 30) and diversity and (recent_success < 0.2)
            if trigger:
                # Perform local search from current best and optionally second best
                candidates = [self.x_opt]
                if pop_size >= 2:
                    candidates.append(pop[np.argmin(fitness)])  # already best, but double check
                # Actually use best and a random elite (second best)
                if pop_size >= 2:
                    sorted_idx = np.argsort(fitness)
                    candidates.append(pop[sorted_idx[1]])

                for x0 in candidates:
                    if evals + 10 >= budget_left:
                        break
                    x = x0.copy()
                    f = func(x)  # we already know f(x0) but budget already counted; avoid duplicate
                    # We'll use SPSA gradient with 2 evaluations
                    c = 1e-3 * domain_range.mean()
                    def spsa_grad(x):
                        delta = np.random.choice([-1, 1], size=dim)
                        x_plus = np.clip(x + c * delta, lb, ub)
                        x_minus = np.clip(x - c * delta, lb, ub)
                        f_plus = func(x_plus)
                        f_minus = func(x_minus)
                        ev = 2
                        if f_plus == np.inf or f_minus == np.inf:
                            return None, None, None, ev
                        g = (f_plus - f_minus) / (2.0 * c) * delta
                        return g, f_plus, f_minus, ev

                    g, _, _, ev_cost = spsa_grad(x)
                    evals += ev_cost
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break

                    # L‑BFGS iterations
                    ls_iters = max(2, min(8, int(0.02 * budget_left / dim)))
                    for it in range(ls_iters):
                        if evals + 3 >= self.budget:
                            break
                        # Two‑loop recursion
                        q = g.copy()
                        alpha = np.zeros(len(s_list))
                        for i in range(len(s_list)-1, -1, -1):
                            s_y = np.dot(s_list[i], y_list[i])
                            if s_y == 0:
                                alpha[i] = 0
                            else:
                                alpha[i] = np.dot(s_list[i], q) / s_y
                            q = q - alpha[i] * y_list[i]
                        d = -q
                        if len(s_list) > 0:
                            s_y_last = np.dot(s_list[-1], y_list[-1])
                            y_y_last = np.dot(y_list[-1], y_list[-1])
                            H0 = s_y_last / (y_y_last + 1e-30)
                            d = H0 * d
                        for i in range(len(s_list)):
                            s_y = np.dot(s_list[i], y_list[i])
                            if s_y == 0:
                                beta = 0
                            else:
                                beta = np.dot(y_list[i], d) / s_y
                            d = d + (alpha[i] - beta) * s_list[i]

                        # Armijo line search (max 5 evaluations)
                        alpha_step = 1.0
                        c_armijo = 1e-4
                        f0 = f
                        x_new = None
                        f_new = None
                        for _ in range(5):
                            x_try = np.clip(x + alpha_step * d, lb, ub)
                            f_try = func(x_try)
                            evals += 1
                            if f_try <= f0 + c_armijo * alpha_step * np.dot(g, x_try - x):
                                x_new = x_try
                                f_new = f_try
                                break
                            alpha_step *= 0.5
                        if x_new is None or alpha_step < 1e-12:
                            break

                        # Compute new gradient for update
                        g_new, _, _, ev2 = spsa_grad(x_new)
                        evals += ev2
                        if g_new is None:
                            break
                        s = x_new - x
                        y = g_new - g
                        s_y = np.dot(s, y)
                        if s_y > 1e-8:
                            if len(s_list) >= L_mem:
                                s_list.pop(0)
                                y_list.pop(0)
                            s_list.append(s.copy())
                            y_list.append(y.copy())
                        x = x_new
                        f = f_new
                        g = g_new

                        if f < self.f_opt:
                            self.f_opt = f
                            self.x_opt = x.copy()

                # Inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # Also inject a slightly perturbed copy
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
                # Update frequency based on improvement
                if self.f_opt < best_old - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.15))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(12, int(0.1 * max_gen)):
                n_restart = max(2, int(0.6 * pop_size))
                # Generate new points: half near best, half from LHS
                perm2 = np.tile(np.arange(1, n_restart+1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs_new = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.15 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + lhs_new[idx] * domain_range
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memories
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt