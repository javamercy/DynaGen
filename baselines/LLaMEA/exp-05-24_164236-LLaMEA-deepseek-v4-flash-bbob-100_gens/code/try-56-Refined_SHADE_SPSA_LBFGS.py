import numpy as np

class Refined_SHADE_SPSA_LBFGS:
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

        # L-SHADE style initial population size (smaller for high dimensions)
        N_init = max(4, int(4 + 3 * np.log(dim)))
        N_min = 4
        pop_size = N_init
        # Estimate max generations (budget / pop_size * some factor)
        max_gen = int(self.budget / (pop_size * 1.2))

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

        # Archive (for current-to-pbest)
        archive = []
        archive_size = 2 * pop_size  # L-SHADE uses 2.6*popsize, but keep moderate

        # Stagnation and local search control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        # Local search frequency: start moderate, adapt based on improvement
        ls_freq = max(5, int(0.04 * max_gen))
        min_freq = 2
        max_freq = max(20, int(0.15 * max_gen))

        # L-BFGS memory
        L_mem = 5
        s_list = []
        y_list = []

        # Success history for LS trigger
        success_history = []

        # Additional variables for adaptive line search
        ls_step_history = [1.0]  # history of last successful step lengths

        while evals < self.budget:
            gen += 1

            # Linear population reduction (L-SHADE style)
            ratio = 1.0 - (gen / max_gen) if max_gen > 0 else 0
            new_pop_size = max(N_min, int(N_init * ratio + N_min * (1 - ratio)))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                # Trim archive to keep archive_size proportional to pop_size
                archive_size = 2 * pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate: increases with generation
            p = 0.15 + 0.35 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                best_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:best_size]
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
                    def get_idx(idx):
                        return pop[idx] if idx < pop_size else archive[idx - pop_size]
                    x_r1 = get_idx(r1)
                    x_r2 = get_idx(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F, CR from memory with adaptive noise
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.05 * np.random.randn()
                CR = mem_CR[r] + 0.05 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1 (sometimes replace with current-to-rand/1 for diversity)
                if np.random.rand() < 0.8:  # 80% current-to-pbest
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                else:  # current-to-rand/1 (more explorative)
                    mutant = pop[i] + F * (x_r1 - pop[i]) + F * (x_r2 - x_pbest)

                # Crossover: binomial (preferred) sometimes exponential
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
                    # Archive update (L-SHADE style: replace random if full)
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

            # Update SHADE memory (weighted Lehmer for F, weighted mean for CR)
            if len(success_F) > 0:
                w = np.array(weight)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    # Lehmer mean for F
                    F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                    CR_mean = np.sum(w * np.array(success_CR))
                    mem_F[mem_idx] = F_lehmer
                    mem_CR[mem_idx] = CR_mean
                    mem_idx = (mem_idx + 1) % mem_size

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_history.append(success_rate)
            if len(success_history) > 10:
                success_history.pop(0)

            # ---------- Adaptive local search (SPSA-based L-BFGS) ----------
            budget_left = self.budget - evals
            # Conditions: generation multiple, budget left, low diversity, low success
            low_success = (len(success_history) < 5 or np.mean(success_history[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.3 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 40 and diversity and low_success):

                c = 0.01 * domain_range.mean() * (1 + 1.0 / dim)  # slightly scaled c
                def spsa_grad(x):
                    delta = np.random.choice([-1, 1], size=dim)
                    x_plus = np.clip(x + c * delta, lb, ub)
                    x_minus = np.clip(x - c * delta, lb, ub)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    if f_plus == np.inf or f_minus == np.inf:
                        return None, None, None
                    g = (f_plus - f_minus) / (2 * c) * delta
                    return g, f_plus, f_minus

                x = self.x_opt.copy()
                f = self.f_opt
                # Number of LS iterations based on remaining budget and dim
                ls_iters = max(2, min(8, int(0.02 * budget_left / dim)))

                for it in range(ls_iters):
                    if evals + 3 >= self.budget:  # need at least 3 evals
                        break
                    g, f_plus, f_minus = spsa_grad(x)
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break
                    evals += 2

                    # L-BFGS two-loop recursion
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for k in range(len(s_list)-1, -1, -1):
                        sy = np.dot(s_list[k], y_list[k])
                        alpha[k] = np.dot(s_list[k], q) / (sy + 1e-30)
                        q = q - alpha[k] * y_list[k]
                    d = -q
                    if len(s_list) > 0:
                        sy_last = np.dot(s_list[-1], y_list[-1])
                        yy_last = np.dot(y_list[-1], y_list[-1])
                        H0 = sy_last / (yy_last + 1e-30)
                        d = H0 * d
                    for k in range(len(s_list)):
                        sy = np.dot(s_list[k], y_list[k])
                        beta = np.dot(y_list[k], d) / (sy + 1e-30)
                        d = d + (alpha[k] - beta) * s_list[k]

                    # Adaptive line search: start from a history-based step length
                    alpha_step = ls_step_history[-1] if len(ls_step_history) > 0 else 1.0
                    c_armijo = 1e-4
                    f0 = f
                    x_new = None
                    f_new = None
                    best_step = None
                    for _ in range(8):  # up to 8 evaluations
                        x_try = np.clip(x + alpha_step * d, lb, ub)
                        f_try = func(x_try)
                        evals += 1
                        if f_try <= f0 + c_armijo * alpha_step * np.dot(g, x_try - x):
                            x_new = x_try
                            f_new = f_try
                            best_step = alpha_step
                            break
                        alpha_step *= 0.5
                    if x_new is None or alpha_step < 1e-12:
                        break
                    # Update step history (smooth)
                    if best_step is not None:
                        ls_step_history.append(best_step)
                        if len(ls_step_history) > 5:
                            ls_step_history.pop(0)

                    # Compute new gradient for L-BFGS update
                    g_new, _, _ = spsa_grad(x_new)
                    evals += 2
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

                # Inject best into population with a few perturbed copies
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # Add a couple of perturbed points if budget allows
                    for _ in range(min(2, budget_left - 2)):
                        if evals >= self.budget:
                            break
                        perturbed = self.x_opt + 0.01 * np.random.randn(dim) * domain_range
                        perturbed = np.clip(perturbed, lb, ub)
                        f_pert = func(perturbed)
                        evals += 1
                        if f_pert < fitness.max():
                            worst_idx = np.argmax(fitness)
                            pop[worst_idx] = perturbed.copy()
                            fitness[worst_idx] = f_pert
                            if f_pert < self.f_opt:
                                self.f_opt = f_pert
                                self.x_opt = perturbed.copy()

                # Adapt local search frequency based on recent improvement
                if f_new is not None and f_new < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.15))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.06 * max_gen)):
                # Restart: keep best, resample 60% of population
                n_restart = max(1, int(0.6 * pop_size))
                # Use a mixture: half near best, half quasi-random
                for idx in range(min(n_restart, pop_size)):
                    if idx < n_restart // 2:
                        scale = 0.05 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        # Simple random uniform (instead of LHS for speed)
                        pop[idx] = lb + np.random.rand(dim) * domain_range
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset SHADE memory, archive, L-BFGS history
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                mem_idx = 0
                archive.clear()
                s_list.clear()
                y_list.clear()
                ls_step_history = [1.0]
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        # Final local search if budget left
        budget_left = self.budget - evals
        if budget_left >= 10:
            c = 0.001 * domain_range.mean()
            x = self.x_opt.copy()
            f = self.f_opt
            ls_iters = min(5, budget_left // 3)
            for it in range(ls_iters):
                if evals + 3 > self.budget:
                    break
                delta = np.random.choice([-1, 1], size=dim)
                x_plus = np.clip(x + c * delta, lb, ub)
                x_minus = np.clip(x - c * delta, lb, ub)
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                evals += 2
                if f_plus == np.inf or f_minus == np.inf:
                    continue
                g = (f_plus - f_minus) / (2 * c) * delta
                if np.linalg.norm(g) < 1e-12:
                    break
                # Simple gradient descent with small step
                alpha = 1.0
                for _ in range(3):
                    x_try = np.clip(x - alpha * g, lb, ub)
                    f_try = func(x_try)
                    evals += 1
                    if f_try < f:
                        x = x_try
                        f = f_try
                        if f < self.f_opt:
                            self.f_opt = f
                            self.x_opt = x.copy()
                        break
                    alpha *= 0.5
                else:
                    break

        return self.f_opt, self.x_opt