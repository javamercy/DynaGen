import numpy as np

class Enhanced_SHADE_Plus:
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

        # Improved initial population: scrambled Sobol-like sequence
        pop_size = max(10, min(50, int(14 * np.sqrt(dim) * (1 + 0.1 * np.log(dim)))))
        # Use a simple Halton-like sequence with scrambled order
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i //= base
                f /= base
            return result
        pop = np.zeros((pop_size, dim))
        seeds = np.random.randint(0, 100000, size=dim)
        for j in range(dim):
            base = primes[j % len(primes)]
            for i in range(pop_size):
                # add random shift for scrambling
                shift = np.random.rand() * 0.1
                pop[i, j] = (halton(i+1, base) + shift) % 1.0
        pop = lb + pop * (ub - lb)

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # SHADE memory for F and CR
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Archive for L-SHADE
        archive = []
        archive_size = pop_size

        # Stagnation detection
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        # Adaptive local search frequency
        ls_freq = max(5, int(0.05 * (self.budget / pop_size)))
        # L-BFGS memory
        L_mem = 7
        s_list = []
        y_list = []

        # Success history for LS trigger
        success_rates = []

        # Budget-aware parameters
        budget_per_gen = self.budget // (pop_size * 2)  # rough generations

        while evals < self.budget:
            gen += 1
            # Linear population size reduction based on remaining budget
            remaining_frac = (self.budget - evals) / self.budget
            new_pop_size = max(4, int(pop_size * remaining_frac))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[:archive_size]

            # Adaptive pbest rate: more exploitative as budget decreases
            p = 0.2 * (1 - remaining_frac) ** 1.2 + 0.1
            p = min(max(p, 0.05), 0.5)

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

                # Select r1, r2 from union of pop and archive (excluding current i)
                union = list(range(pop_size)) + list(range(len(archive)))
                # Build efficient index mapping to avoid recomputation
                # Simple approach: remove i if in pop
                if i < pop_size:
                    union = [j for j in union if j != i]
                else:
                    union = list(union)  # copy
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

                # Sample F, CR from memory with small noise
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 0.9)  # narrow range to avoid extremes
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

            # Update SHADE memory using weighted Lehmer mean for F and weighted mean for CR
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
            # Trigger if stagnation, low diversity, and enough budget left
            diversity = np.std(pop, axis=0).mean()
            low_success = len(success_rates) > 3 and np.mean(success_rates[-3:]) < 0.15
            if (gen % ls_freq == 0 and budget_left > 10 * dim and
                diversity < 0.5 * (ub - lb).mean() and (stagnation_counter > 3 or low_success)):

                # Number of LS iterations based on remaining budget
                ls_iters = max(2, min(15, int(0.05 * budget_left / dim)))
                # SPSA gradient with adaptive perturbation
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
                # L-BFGS two-loop recursion with SPSA gradient
                for it in range(ls_iters):
                    if evals + 2 > self.budget:
                        break
                    # Compute gradient
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
                    # Line search (Armijo) using few evaluations
                    alpha_step = 1.0
                    c_armijo = 1e-4
                    f0 = f
                    best_found = False
                    for _ in range(5):  # reduced steps
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= f0 + c_armijo * alpha_step * np.dot(g, x_new - x):
                            best_found = True
                            break
                        alpha_step *= 0.5
                    if not best_found:
                        break
                    # Update L-BFGS memory
                    s = x_new - x
                    # Compute new gradient at x_new (SPSA)
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
                # Inject best local point into population
                if self.f_opt < fitness.max() and evals < self.budget:
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # Perturb a second point for diversity
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
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.08 * budget_per_gen)):
                n_restart = max(1, int(0.3 * pop_size))
                # Better restart: mix around best and random quasi-random points
                # Generate Sobol-like points for randomness
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * (ub - lb) * (1 - remaining_frac) + 0.01 * (ub - lb)
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
                ls_freq = min(budget_per_gen // 2, ls_freq + 2)

        # Final local search if budget remains
        budget_left = self.budget - evals
        if budget_left > dim:
            # Simple greedy descent: random perturbations around best
            for _ in range(min(budget_left // dim, 5)):
                if evals >= self.budget:
                    break
                x_new = self.x_opt + 0.001 * np.random.randn(dim) * (ub - lb)
                x_new = np.clip(x_new, lb, ub)
                f_new = func(x_new)
                evals += 1
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = x_new.copy()

        return self.f_opt, self.x_opt