import numpy as np

class Enhanced_SHADE_LS:
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

        # Population sizing (L-SHADE style)
        N_init = max(10, int(18 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_evals = self.budget

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

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

        # Archive
        archive = []
        archive_size = N_init

        # Stagnation tracking
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search control
        ls_freq = max(8, int(0.04 * self.budget / pop_size))
        min_freq = 3
        max_freq = max(30, int(0.15 * self.budget / pop_size))
        ls_stagnation_threshold = max(5, int(0.03 * self.budget / pop_size))

        # L‑BFGS memory
        L_mem = 10
        s_list = []
        y_list = []

        # Success history for LS trigger
        success_rates = []

        while evals < max_evals:
            gen += 1

            # Linear population reduction (L-SHADE)
            if pop_size > N_min:
                ratio = 1.0 - (gen * pop_size) / max_evals  # approximate generation count
                # more exact: use evals proportion
                # Actually L-SHADE uses a target generation count: but we will use evals proportion
                # We'll approximate based on evals:
                evals_left = max_evals - evals
                max_possible_gen = max_evals / pop_size  # rough
                # simple linear reduction over evals
                fraction_used = evals / max_evals
                new_pop_size = max(N_min, int(N_init - (N_init - N_min) * fraction_used))
                if new_pop_size < pop_size:
                    idx_sorted = np.argsort(fitness)
                    pop = pop[idx_sorted[:new_pop_size]].copy()
                    fitness = fitness[idx_sorted[:new_pop_size]]
                    pop_size = new_pop_size
                    if len(archive) > archive_size:
                        np.random.shuffle(archive)
                        archive = archive[:archive_size]

            # pbest rate: inversely proportional to pop size (L-SHADE)
            p = 0.2 * (N_init / pop_size) if pop_size > 0 else 0.5
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= max_evals:
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

                # Sample F, CR from memory
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
                    # Archive replacement
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

            # Update SHADE memory (weighted Lehmer mean for F, arithmetic for CR)
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

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Adaptive local search (SPSA‑based L‑BFGS) ----------
            budget_left = self.budget - evals
            # Conditions: stagnation, low diversity, low success rate, and sufficient budget
            stagnation_flag = stagnation_counter > ls_stagnation_threshold
            low_success = (len(success_rates) >= 5 and np.mean(success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.3 * np.mean(domain_range)
            if (stagnation_flag or (gen % ls_freq == 0)) and budget_left > 30 and low_success and diversity:
                # SPSA gradient estimation (2 evaluations per direction, average 2 directions = 4 evaluations)
                c = 1e-3 * domain_range.mean()
                def spsa_grad(x):
                    # Two random directions averaged
                    g = np.zeros(dim)
                    f_vals = []
                    for _ in range(2):
                        delta = np.random.choice([-1, 1], size=dim)
                        x_plus = np.clip(x + c * delta, lb, ub)
                        x_minus = np.clip(x - c * delta, lb, ub)
                        f_plus = func(x_plus)
                        f_minus = func(x_minus)
                        # Check for infinite
                        if np.isinf(f_plus) or np.isinf(f_minus):
                            return None, None, None
                        g += (f_plus - f_minus) / (2 * c) * delta
                        f_vals.extend([f_plus, f_minus])
                    g /= 2.0
                    return g, f_vals[0], f_vals[1]  # return last two for line search?

                x = self.x_opt.copy()
                f = self.f_opt
                ls_iters = max(2, min(10, int(0.03 * budget_left / dim)))
                # reduce iterations if too expensive
                for it in range(ls_iters):
                    if evals + 6 >= self.budget:  # need up to 6 evals per iteration
                        break
                    g, _, _ = spsa_grad(x)
                    evals += 4
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break

                    # L‑BFGS two‑loop recursion
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

                    # Quadratic line search (Armijo with quadratic fit)
                    alpha_step = 1.0
                    c_armijo = 1e-4
                    f0 = f
                    g0 = g
                    # Evaluate at alpha=1 and alpha=0.5 for quadratic fit
                    x1 = np.clip(x + alpha_step * d, lb, ub)
                    f1 = func(x1)
                    evals += 1
                    if f1 <= f0 + c_armijo * alpha_step * np.dot(g0, x1 - x):
                        x_new = x1
                        f_new = f1
                    else:
                        alpha_half = 0.5
                        x_half = np.clip(x + alpha_half * d, lb, ub)
                        f_half = func(x_half)
                        evals += 1
                        if f_half <= f0 + c_armijo * alpha_half * np.dot(g0, x_half - x):
                            x_new = x_half
                            f_new = f_half
                        else:
                            # Quadratic fit: a * alpha^2 + b * alpha + c
                            # Use points (0, f0), (0.5, f_half), (1, f1)
                            a = 2*(f0 - 2*f_half + f1)
                            b = 4*f_half - 3*f0 - f1
                            if a > 0:
                                alpha_opt = -b / (2*a)
                                alpha_opt = np.clip(alpha_opt, 0.1, 0.9)
                                x_opt = np.clip(x + alpha_opt * d, lb, ub)
                                f_opt_ls = func(x_opt)
                                evals += 1
                                if f_opt_ls < f0 and f_opt_ls < f_half and f_opt_ls < f1:
                                    x_new = x_opt
                                    f_new = f_opt_ls
                                else:
                                    # fallback to best of three
                                    vals = [(f0, x), (f_half, x_half), (f1, x1)]
                                    vals.sort(key=lambda v: v[0])
                                    x_new, f_new = vals[0][1], vals[0][0]
                                    if x_new is x:
                                        break  # no progress
                            else:
                                # fallback to best of three
                                vals = [(f0, x), (f_half, x_half), (f1, x1)]
                                vals.sort(key=lambda v: v[0])
                                x_new, f_new = vals[0][1], vals[0][0]
                                if x_new is x:
                                    break

                    # Compute new gradient for L‑BFGS update
                    g_new, _, _ = spsa_grad(x_new)
                    evals += 4
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
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()

                # Inject best into population with a few mutated copies
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # add a few perturbed copies
                    for _ in range(min(3, pop_size//4)):
                        if evals >= self.budget:
                            break
                        perturbed = self.x_opt + 0.01 * np.random.randn(dim) * domain_range * (1 + 0.1*np.random.randn())
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
                if f < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(15, int(0.1 * self.budget / pop_size)):
                # Restart: keep best 20% of population, replace rest
                sort_idx = np.argsort(fitness)
                keep = max(2, pop_size // 5)
                new_pop = pop[sort_idx[:keep]].copy()
                new_fit = fitness[sort_idx[:keep]].copy()
                # Generate rest: near best with decreasing radius
                radius = 0.1 * domain_range * (1 - evals / self.budget) ** 0.5 + 0.01
                for _ in range(pop_size - keep):
                    if evals >= self.budget:
                        break
                    candidate = self.x_opt + radius * np.random.randn(dim)
                    candidate = np.clip(candidate, lb, ub)
                    new_pop = np.vstack((new_pop, candidate))
                    f_cand = func(candidate)
                    evals += 1
                    new_fit = np.append(new_fit, f_cand)
                    if f_cand < self.f_opt:
                        self.f_opt = f_cand
                        self.x_opt = candidate.copy()
                pop = new_pop
                fitness = new_fit
                pop_size = len(pop)
                # Reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt