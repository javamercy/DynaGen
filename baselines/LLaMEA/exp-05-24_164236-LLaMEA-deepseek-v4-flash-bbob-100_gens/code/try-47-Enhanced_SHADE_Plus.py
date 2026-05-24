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

        # Latin Hypercube Sampling
        def lhs(n, d):
            x = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                x[:, j] = (perm + np.random.uniform(0,1,size=n)) / n
            return x

        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Initial population via LHS
        sobol = lhs(pop_size, dim)
        pop = lb + sobol * (ub - lb)

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
        archive_size = pop_size

        # Stagnation
        best_old = self.f_opt
        stagnation_counter = 0
        stagnation_threshold = max(5, int(0.05 * max_gen))

        # Local search state
        ls_freq_init = max(8, int(0.04 * max_gen))
        ls_freq = ls_freq_init
        # Barzilai-Borwein constants
        alpha_bb = 1e-3
        s_list = []
        y_list = []
        L_mem = 7
        prev_grad = None
        prev_x = None

        # Success rates for LS trigger
        success_rates = []
        best_improve_window = []

        for gen in range(1, max_gen + 1):
            if evals >= self.budget:
                break

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / (1.5 * max_gen)))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Adaptive p-best: start high, decay exponentially
            p = 0.5 * np.exp(-2.0 * gen / max_gen) + 0.1
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

                # Select r1, r2 from union of pop and archive
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    x_r1 = pop[r1] if r1 < pop_size else archive[r1 - pop_size]
                    x_r2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                else:
                    idx = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(idx, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F, CR
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Binomial crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Archive update
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
            if success_F:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # Track success and improvement
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)
            if self.f_opt < best_old:
                best_improve_window.append(best_old - self.f_opt)
                best_old = self.f_opt
            if len(best_improve_window) > 10:
                best_improve_window.pop(0)

            # Local search trigger
            budget_left = self.budget - evals
            std_fit = np.std(fitness)
            low_success = (len(success_rates) >= 5 and np.mean(success_rates[-5:]) < 0.15)
            early_improvement = (len(best_improve_window) < 5 or np.mean(best_improve_window) > 1e-5)

            if (gen % ls_freq == 0 and budget_left > 30 and std_fit < 0.5 and low_success and early_improvement):
                # Local search using SPSA + Barzilai-Borwein L-BFGS
                max_ls_iters = min(10, max(2, int(0.02 * budget_left / (dim + 1))))
                c = 1e-3 * (ub - lb).mean()
                x = self.x_opt.copy()
                f = self.f_opt

                # Reset L-BFGS memory if started from new point
                if prev_grad is None or not np.allclose(x, prev_x):
                    s_list.clear()
                    y_list.clear()
                    # Compute initial gradient
                    delta = np.random.choice([-1, 1], size=dim)
                    xp = np.clip(x + c*delta, lb, ub)
                    xm = np.clip(x - c*delta, lb, ub)
                    fp = func(xp); fm = func(xm)
                    evals += 2
                    g = (fp - fm) / (2*c) * (1.0/delta)
                    prev_grad = g.copy()
                    prev_x = x.copy()
                else:
                    g = prev_grad.copy()

                for it in range(max_ls_iters):
                    if evals + 2 >= self.budget:
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
                    if s_list:
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

                    # Barzilai-Borwein step size as starting point (if available)
                    if s_list:
                        # approximate step = (s'*s)/(s'*y) or (s'*y)/(y'*y)
                        s = s_list[-1]; y = y_list[-1]
                        sy = np.dot(s, y); ss = np.dot(s, s)
                        if sy > 0:
                            alpha_bb = ss / sy
                        else:
                            alpha_bb = 1e-3
                    else:
                        alpha_bb = 1e-3
                    alpha_step = max(1e-8, min(1.0, alpha_bb))

                    # Simple line search (constant step)
                    x_new = np.clip(x + alpha_step * d, lb, ub)
                    f_new = func(x_new)
                    evals += 1
                    # Accept if improvement
                    if f_new <= f:
                        # Update L-BFGS with new gradient
                        delta2 = np.random.choice([-1,1], size=dim)
                        xp2 = np.clip(x_new + c*delta2, lb, ub)
                        xm2 = np.clip(x_new - c*delta2, lb, ub)
                        fp2 = func(xp2); fm2 = func(xm2)
                        evals += 2
                        g_new = (fp2 - fm2) / (2*c) * (1.0/delta2)
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
                        g = g_new
                        prev_grad = g.copy()
                        prev_x = x.copy()
                        if f < self.f_opt:
                            self.f_opt = f
                            self.x_opt = x.copy()
                    else:
                        break

                # Inject best and a perturbed point
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    if evals < self.budget:
                        perturb = self.x_opt + 0.01 * np.random.randn(dim) * (ub - lb)
                        perturb = np.clip(perturb, lb, ub)
                        fp = func(perturb)
                        evals += 1
                        if fp < fitness.max():
                            worst2 = np.argmax(fitness)
                            pop[worst2] = perturb
                            fitness[worst2] = fp
                            if fp < self.f_opt:
                                self.f_opt = fp
                                self.x_opt = perturb.copy()

            # Stagnation detection and full restart
            if self.f_opt >= best_old - 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_old = self.f_opt

            if stagnation_counter >= stagnation_threshold and budget_left > 50:
                # Full restart: new population centered on best, plus uniform sampling
                new_pop_size = max(N_min, int(0.8 * pop_size))
                new_pop = np.empty((new_pop_size, dim))
                n_best = max(2, new_pop_size // 2)
                # Generate around best with decreasing scale as budget runs low
                scale = max(0.01, (ub - lb).mean() * (1 - gen / max_gen))
                for i in range(n_best):
                    new_pop[i] = self.x_opt + np.random.randn(dim) * scale
                # Remaining from uniform LHS
                lhs_rest = lhs(new_pop_size - n_best, dim)
                for i in range(new_pop_size - n_best):
                    new_pop[n_best+i] = lb + lhs_rest[i] * (ub - lb)
                new_pop = np.clip(new_pop, lb, ub)
                new_fitness = np.empty(new_pop_size)
                for i in range(new_pop_size):
                    new_fitness[i] = func(new_pop[i])
                    evals += 1
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_pop[i].copy()
                pop = new_pop
                fitness = new_fitness
                pop_size = new_pop_size
                # Reset memories
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                s_list.clear()
                y_list.clear()
                prev_grad = None
                prev_x = None
                archive.clear()
                stagnation_counter = 0
                # Increase LS frequency slightly
                ls_freq = min(max_gen // 3, ls_freq + 2)

        # Final local search if budget remains
        while evals < self.budget:
            x = self.x_opt
            delta = np.random.choice([-1,1], size=dim)
            c = 1e-3 * (ub - lb).mean()
            xp = np.clip(x + c*delta, lb, ub)
            xm = np.clip(x - c*delta, lb, ub)
            fp = func(xp); fm = func(xm)
            evals += 2
            if evals >= self.budget:
                break
            g = (fp - fm) / (2*c) * (1.0/delta)
            step = 0.01
            x_new = np.clip(x - step*g, lb, ub)
            f_new = func(x_new)
            evals += 1
            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new.copy()
            else:
                break

        return self.f_opt, self.x_opt