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
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # initial population (Sobol-like)
        sobol = np.random.rand(pop_size, dim)
        for j in range(dim):
            sobol[:, j] = (np.argsort(sobol[:, j]) + 0.5) / pop_size
        pop = lb + sobol * (ub - lb)

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
        stagnation_counter = 0
        gen = 0

        # mode probabilities (0: current-to-pbest, 1: rand/2, 2: best/1)
        mode_probs = np.array([0.6, 0.2, 0.2])
        mode_success = np.ones(3)

        success_rates = []
        ls_freq = max(12, int(0.08 * max_gen))

        # local search: random direction quadratic fit
        def quadratic_line_search(x, f_x, d):
            # bracket minimum using two points along d
            alpha = 1.0
            x1 = np.clip(x + alpha * d, lb, ub)
            f1 = func(x1)
            evals_local = 1
            x2 = np.clip(x - alpha * d, lb, ub)
            f2 = func(x2)
            evals_local += 1
            # quadratic fit
            if f1 < f_x and f2 < f_x:
                # both better
                best = min((f_x, x, 0), (f1, x1, alpha), (f2, x2, -alpha),
                           key=lambda t: t[0])
                return best[1], best[0], evals_local
            elif f1 < f_x:
                # search further in positive direction
                for _ in range(3):
                    alpha *= 2
                    x1 = np.clip(x + alpha * d, lb, ub)
                    f1 = func(x1)
                    evals_local += 1
                    if f1 >= f_x:
                        break
                # use these three points: x (0), x+alpha*d (alpha), x+alpha/2*d?
                return self._quad_fit(x, f_x, x1, f1, d, alpha, 0, evals_local)
            elif f2 < f_x:
                for _ in range(3):
                    alpha *= 2
                    x2 = np.clip(x - alpha * d, lb, ub)
                    f2 = func(x2)
                    evals_local += 1
                    if f2 >= f_x:
                        break
                return self._quad_fit(x, f_x, x2, f2, d, -alpha, 0, evals_local)
            else:
                # no improvement: try smaller step
                for _ in range(5):
                    alpha /= 2
                    if alpha < 1e-8:
                        break
                    x1 = np.clip(x + alpha * d, lb, ub)
                    f1 = func(x1)
                    evals_local += 1
                    x2 = np.clip(x - alpha * d, lb, ub)
                    f2 = func(x2)
                    evals_local += 1
                    if f1 < f_x or f2 < f_x:
                        break
                if f1 < f_x:
                    return self._quad_fit(x, f_x, x1, f1, d, alpha, 0, evals_local)
                elif f2 < f_x:
                    return self._quad_fit(x, f_x, x2, f2, d, -alpha, 0, evals_local)
                else:
                    return x, f_x, evals_local

        def _quad_fit(self, x0, f0, x1, f1, d, a1, a0, evals):
            # use three points: x0 (a0), x1 (a1), and midpoint (a0+a1)/2
            a2 = (a0 + a1) / 2
            x2 = np.clip(x0 + a2 * d, lb, ub)
            f2 = func(x2)
            evals += 1
            # solve quadratic
            A = np.array([[a0**2, a0, 1],
                          [a1**2, a1, 1],
                          [a2**2, a2, 1]])
            coeff = np.linalg.solve(A, np.array([f0, f1, f2]))
            if coeff[0] > 0:
                alpha_min = -coeff[1] / (2 * coeff[0])
                alpha_min = np.clip(alpha_min, a0, a1)
                x_min = np.clip(x0 + alpha_min * d, lb, ub)
                f_min = func(x_min)
                evals += 1
                if f_min < min(f0, f1, f2):
                    return x_min, f_min, evals
            # fallback to best of three
            best_idx = np.argmin([f0, f1, f2])
            best_x = [x0, x1, x2][best_idx]
            best_f = [f0, f1, f2][best_idx]
            return best_x, best_f, evals

        while evals < self.budget:
            gen += 1

            # linear population reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / (1.5 * max_gen)))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            p = 0.2 * (gen / max_gen) ** 1.5 + 0.1
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # adaptive mutation selection
                mode = np.random.choice(3, p=mode_probs)
                if mode == 0:  # current-to-pbest/1
                    pbest_size = max(2, int(p * pop_size))
                    best_indices = np.argsort(fitness)[:pbest_size]
                    pbest_idx = np.random.choice(best_indices)
                    x_pbest = pop[pbest_idx]
                    union = list(range(pop_size)) + list(range(len(archive)))
                    union.remove(i)
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
                    mutant = pop[i] + 0.5 * (x_pbest - pop[i]) + 0.5 * (x_r1 - x_r2)
                elif mode == 1:  # rand/2
                    indices = [j for j in range(pop_size) if j != i]
                    if len(indices) >= 4:
                        r0, r1, r2, r3 = np.random.choice(indices, 4, replace=False)
                        mutant = pop[r0] + 0.5 * (pop[r1] - pop[r2]) + 0.5 * (pop[r3] - pop[r0])
                    else:
                        mutant = pop[i] + 0.5 * (pop[r0] - pop[r1]) + 0.1 * (np.random.randn(dim))
                else:  # best/1
                    best_idx = np.argmin(fitness)
                    indices = [j for j in range(pop_size) if j != best_idx]
                    if len(indices) >= 2:
                        r1, r2 = np.random.choice(indices, 2, replace=False)
                        mutant = pop[best_idx] + 0.5 * (pop[r1] - pop[r2])
                    else:
                        mutant = pop[best_idx] + 0.1 * np.random.randn(dim)

                # F and CR
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # crossover (binomial or exponential)
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
                    # reward used mode
                    mode_success[mode] += 0.1
                else:
                    # penalize mode
                    mode_success[mode] -= 0.05

            # update mode probabilities
            mode_probs = np.abs(mode_success) / np.sum(np.abs(mode_success))
            mode_probs = np.clip(mode_probs, 0.05, 0.9)
            mode_probs /= mode_probs.sum()

            # update SHADE memory
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Local search phase (quadratic line search along random directions) ----------
            if (gen % ls_freq == 0 and (self.budget - evals) > 30 and
                np.std(fitness) < 0.8 and
                np.mean(success_rates[-5:]) < 0.2):
                # perform local search from best point
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                ls_iters = max(2, min(10, int(0.03 * (self.budget - evals) / (dim + 1))))
                for _ in range(ls_iters):
                    if evals + 5 >= self.budget:
                        break
                    # random direction
                    d = np.random.randn(dim)
                    d = d / (np.linalg.norm(d) + 1e-30)
                    x_new, f_new, used = quadratic_line_search(x_best, f_best, d)
                    evals += used
                    if f_new < f_best:
                        x_best, f_best = x_new, f_new
                        if f_new < self.f_opt:
                            self.f_opt = f_new
                            self.x_opt = x_new.copy()
                # inject results into population
                if f_best < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = x_best
                    fitness[worst] = f_best
                    # also add a perturbed copy if budget permits
                    if evals < self.budget:
                        perturb = x_best + 0.02 * np.random.randn(dim) * (ub - lb)
                        perturb = np.clip(perturb, lb, ub)
                        f_pert = func(perturb)
                        evals += 1
                        if f_pert < fitness.max():
                            worst2 = np.argmax(fitness)
                            pop[worst2] = perturb
                            fitness[worst2] = f_pert
                            if f_pert < self.f_opt:
                                self.f_opt = f_pert
                                self.x_opt = perturb.copy()

            # ---------- Stagnation / restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.15 * max_gen)):
                n_restart = max(2, int(0.6 * pop_size))
                # generate new points around best with scaled covariance
                cov = np.diag((np.std(pop, axis=0) + 1e-6) ** 2)
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    L = np.diag(np.std(pop, axis=0) + 1e-6)
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.2 * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + scale * np.dot(L, np.random.randn(dim))
                    else:
                        pop[idx] = np.random.uniform(lb, ub, dim)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                stagnation_counter = 0
                ls_freq = min(max_gen // 3, ls_freq + 3)

        return self.f_opt, self.x_opt