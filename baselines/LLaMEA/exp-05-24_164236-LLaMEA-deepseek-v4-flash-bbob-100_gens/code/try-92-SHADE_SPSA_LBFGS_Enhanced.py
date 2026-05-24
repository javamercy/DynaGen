import numpy as np

class SHADE_SPSA_LBFGS_Enhanced:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        dim = self.dim
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        domain_range = ub - lb
        n_perturb = 3  # number of SPSA perturbations to average
        L_mem = 5      # L-BFGS memory size

        # L-SHADE parameters: linear population reduction
        N_init = max(8, int(15 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)
        # SHADE memory
        mem_size = 10
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin hypercube initial population
        n_lhs = pop_size
        perm = np.tile(np.arange(1, n_lhs + 1), (dim, 1)).T
        for j in range(dim):
            perm[:, j] = np.random.permutation(perm[:, j])
        sobol = (perm - 0.5) / n_lhs
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
        best_old = self.f_opt
        stagnation = 0

        # LS trigger parameters
        ls_freq = max(8, int(0.05 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))
        success_rates = []
        s_list = []
        y_list = []

        while evals < self.budget:
            # Linear population reduction
            ratio = 1.0 - evals / self.budget
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx = np.argsort(fitness)
                pop = pop[idx[:new_pop_size]].copy()
                fitness = fitness[idx[:new_pop_size]]
                pop_size = new_pop_size
                archive = archive[:archive_size] if len(archive) > archive_size else archive

            # pbest rate: linear growth
            p = 0.1 + 0.4 * (evals / self.budget)

            success_F = []
            success_CR = []
            weights = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # r1, r2 from pop + archive
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

                # Sample F, CR from memory with noise
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # current-to-pbest/1 mutation
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Binomial crossover (70%) or exponential (30%)
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
                    idx_exp = (np.arange(dim) + start) % dim
                    mask = np.zeros(dim, dtype=bool)
                    mask[idx_exp[:L]] = True
                    trial = np.where(mask, mutant, pop[i])

                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        archive[np.random.randint(len(archive))] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weights.append(max(imp, 1e-12))
                    n_success += 1

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update SHADE memory
            if len(success_F) > 0:
                w = np.array(weights)
                w_sum = w.sum()
                if w_sum > 0:
                    w /= w_sum
                    F_lehmer = (w * np.array(success_F)**2).sum() / ((w * np.array(success_F)).sum() + 1e-30)
                    F_lehmer = np.clip(F_lehmer, 0.1, 1.0)
                    CR_mean = (w * np.array(success_CR)).sum()
                    CR_mean = np.clip(CR_mean, 0.0, 1.0)
                    mem_F[mem_idx] = F_lehmer
                    mem_CR[mem_idx] = CR_mean
                    mem_idx = (mem_idx + 1) % mem_size

            # Success rate monitoring for LS
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ----- Adaptive local search (improved SPSA + L-BFGS) -----
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.5 * domain_range.mean()
            if (evals % ls_freq == 0 and budget_left > 50 and diversity and low_success):
                # Multi-perturbation SPSA gradient estimation (averaged)
                x = self.x_opt.copy()
                f = self.f_opt
                ls_iters = min(8, max(3, int(0.03 * budget_left / dim)))
                c = 1e-3 * domain_range.mean()

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

                # Average over n_perturb random perturbations
                def avg_grad(x):
                    g_list = []
                    for _ in range(n_perturb):
                        g, _, _ = spsa_grad(x)
                        if g is not None:
                            g_list.append(g)
                    if len(g_list) == 0:
                        return None
                    return np.mean(g_list, axis=0)

                for it in range(ls_iters):
                    if evals + 2 * n_perturb + 3 >= self.budget:
                        break
                    # Compute average gradient using n_perturb evaluations
                    g = avg_grad(x)
                    if g is None or np.linalg.norm(g) < 1e-12:
                        break
                    evals += 2 * n_perturb

                    # L-BFGS two-loop recursion
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for k in range(len(s_list)-1, -1, -1):
                        sy = np.dot(s_list[k], y_list[k])
                        if sy == 0:
                            alpha[k] = 0
                        else:
                            alpha[k] = np.dot(s_list[k], q) / sy
                        q -= alpha[k] * y_list[k]
                    d = -q
                    if len(s_list) > 0:
                        sy_last = np.dot(s_list[-1], y_list[-1])
                        yy_last = np.dot(y_list[-1], y_list[-1])
                        H0 = sy_last / (yy_last + 1e-30)
                        d *= H0
                    for k in range(len(s_list)):
                        sy = np.dot(s_list[k], y_list[k])
                        if sy == 0:
                            beta = 0
                        else:
                            beta = np.dot(y_list[k], d) / sy
                        d += (alpha[k] - beta) * s_list[k]

                    # Backtracking line search with quadratic interpolation
                    alpha_step = 1.0
                    # Try to predict step using quadratic fit
                    f0 = f
                    f1_guess = None
                    g_dot_d = np.dot(g, d)
                    if g_dot_d > 0:
                        d = -d
                        g_dot_d = -g_dot_d
                    # Evaluate at alpha=0 (already f0) and alpha=1
                    x1 = np.clip(x + alpha_step * d, lb, ub)
                    f1 = func(x1)
                    evals += 1
                    if f1 <= f0 + 1e-4 * alpha_step * g_dot_d:
                        # Accept step
                        x_new = x1
                        f_new = f1
                        alpha_used = alpha_step
                    else:
                        # Quadratic interpolation: fit parabola through (0, f0), (alpha0, f0+? ), (alpha1, f1)
                        # Use step halving up to 5 trials
                        for _ in range(5):
                            alpha_step *= 0.5
                            x_try = np.clip(x + alpha_step * d, lb, ub)
                            f_try = func(x_try)
                            evals += 1
                            if f_try <= f0 + 1e-4 * alpha_step * g_dot_d:
                                x_new, f_new, alpha_used = x_try, f_try, alpha_step
                                break
                        else:
                            # No progress, break
                            break

                    # Compute new gradient for L-BFGS update
                    g_new = avg_grad(x_new)
                    if g_new is None:
                        break
                    evals += 2 * n_perturb
                    s = x_new - x
                    y = g_new - g
                    sy = np.dot(s, y)
                    if sy > 1e-10:
                        if len(s_list) >= L_mem:
                            s_list.pop(0)
                            y_list.pop(0)
                        s_list.append(s.copy())
                        y_list.append(y.copy())
                    x, f = x_new, f_new
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()
                    # Stop if step size is too small
                    if np.linalg.norm(s) < 1e-8 * domain_range.mean():
                        break

                # Inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # Add a perturbed copy
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

                # Adapt LS frequency
                if f < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # ----- Stagnation detection and restart -----
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > max(10, int(0.08 * max_gen)):
                n_restart = max(2, int(0.5 * pop_size))
                # Reinitialize half near best, half with LHS
                perm2 = np.tile(np.arange(1, n_restart+1), (dim,1)).T
                for j in range(dim):
                    perm2[:,j] = np.random.permutation(perm2[:,j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.2 * domain_range * (1 - evals/self.budget) + 0.01
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
                stagnation = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt