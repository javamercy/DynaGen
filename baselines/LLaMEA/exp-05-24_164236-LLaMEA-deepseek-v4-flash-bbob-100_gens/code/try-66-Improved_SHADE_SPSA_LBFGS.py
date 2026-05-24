import numpy as np

class Improved_SHADE_SPSA_LBFGS:
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

        # L-SHADE parameters
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)  # approximate generations

        # Memory for F and CR
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

        # Archive for mutation
        archive = []
        archive_size = int(2.6 * pop_size)

        # L-BFGS memory (short)
        L_mem = 4
        s_list = []
        y_list = []

        # Control for local search
        ls_interval = max(8, int(0.04 * max_gen))
        min_ls_interval = 4
        max_ls_interval = max(30, int(0.2 * max_gen))
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        success_rates = []

        while evals < self.budget:
            gen += 1

            # Linear population reduction
            ratio = 1.0 - gen / max_gen
            new_pop_size = max(N_min, int(N_init + (N_min - N_init) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[:archive_size]

            # pbest rate: grows with gen
            p = 0.1 + 0.4 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest index
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Select r1, r2 from union of pop and archive
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

                # Sample F and CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial (70%) or exponential (30%)
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
                    # Add to archive
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

            # Success rate monitoring
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Local search (SPSA-based quasi-Newton) ----------
            if (gen % ls_interval == 0 and evals < self.budget - 20):
                # Trigger conditions: low success rate and low diversity
                avg_success = np.mean(success_rates[-5:]) if len(success_rates) >= 5 else 0.0
                diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
                if avg_success < 0.2 and diversity:
                    # SPSA gradient estimation
                    c = 1e-3 * domain_range.mean()
                    def spsa_grad(x):
                        delta = np.random.choice([-1, 1], size=dim)
                        x_plus = np.clip(x + c * delta, lb, ub)
                        x_minus = np.clip(x - c * delta, lb, ub)
                        f_plus = func(x_plus)
                        f_minus = func(x_minus)
                        if np.isinf(f_plus) or np.isinf(f_minus):
                            return None, None, None
                        g = (f_plus - f_minus) / (2.0 * c) * delta
                        return g, f_plus, f_minus

                    x = self.x_opt.copy()
                    f = self.f_opt
                    # Number of LS iterations: limited by remaining budget
                    budget_left = self.budget - evals
                    ls_iters = max(1, min(8, int(0.02 * budget_left / dim)))

                    # Step size for direction (adaptive)
                    step = 0.2 * domain_range.mean()
                    step_decay = 0.5
                    step_gain = 1.2

                    for ls_it in range(ls_iters):
                        if evals + 4 >= self.budget:
                            break

                        g, f_plus, f_minus = spsa_grad(x)
                        if g is None or np.linalg.norm(g) < 1e-12:
                            break
                        evals += 2

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

                        # Simple line search with step adaptation (no extra evals)
                        trial_dir = np.clip(x + step * d, lb, ub)
                        f_trial = func(trial_dir)
                        evals += 1
                        if f_trial < f:
                            # Accept and increase step
                            x = trial_dir
                            f = f_trial
                            step = step * step_gain
                            if f < self.f_opt:
                                self.f_opt = f
                                self.x_opt = x.copy()
                        else:
                            # Try opposite direction
                            trial_opp = np.clip(x - step * d, lb, ub)
                            f_opp = func(trial_opp)
                            evals += 1
                            if f_opp < f:
                                x = trial_opp
                                f = f_opp
                                step = step * step_gain
                                if f < self.f_opt:
                                    self.f_opt = f
                                    self.x_opt = x.copy()
                            else:
                                step = step * step_decay

                        # Update L-BFGS memory with gradient difference
                        g_new, _, _ = spsa_grad(x)
                        evals += 2
                        if g_new is not None:
                            s = x - x_old if 'x_old' in locals() else np.zeros(dim)
                            if np.linalg.norm(s) > 0:
                                y = g_new - g
                                sy = np.dot(s, y)
                                if sy > 1e-10:
                                    if len(s_list) >= L_mem:
                                        s_list.pop(0)
                                        y_list.pop(0)
                                    s_list.append(s.copy())
                                    y_list.append(y.copy())
                        x_old = x.copy()

                    # Inject best back into population (if better)
                    if self.f_opt < fitness.max():
                        worst = np.argmax(fitness)
                        pop[worst] = self.x_opt.copy()
                        fitness[worst] = self.f_opt
                    # Update LS interval adaptively
                    if f < best_old - 1e-8:
                        ls_interval = max(min_ls_interval, int(ls_interval * 0.8))
                    else:
                        ls_interval = min(max_ls_interval, int(ls_interval * 1.1))

            # ---------- Stagnation restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                # Generate new points: half near best, half LHS
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
                # Reset memories
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_interval = min_ls_interval + 2

        return self.f_opt, self.x_opt