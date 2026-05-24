import numpy as np

class Refined_SHADE_Pro:
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

        # Population size: start with larger size, reduce sigmoidally
        N_init = max(8, int(16 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size)  # approximate generations

        # SHADE memory (F and CR)
        mem_size = 8
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Quasi-Sobol like initialization (LHS)
        # Use Sobol-like shuffling for better space coverage
        init = np.empty((pop_size, dim))
        for j in range(dim):
            perm = np.random.permutation(pop_size)
            init[:, j] = lb[j] + (perm + 0.5) / pop_size * (ub[j] - lb[j])
        pop = np.clip(init, lb, ub)

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
        archive_size = int(2.0 * pop_size)  # dynamic factor

        # Stagnation detection
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters
        ls_freq_init = max(8, int(0.05 * max_gen))
        ls_freq = ls_freq_init
        ls_max_iter = max(3, min(8, int(0.02 * (self.budget / (dim + 5)))))
        ls_min_budget = dim * 5 + 20
        # L-BFGS memory
        L_mem = 10
        s_list = []
        y_list = []

        # Success history for adaptation
        success_rates = []

        # Main loop
        while evals < self.budget:
            gen += 1

            # Sigmoidal population size reduction
            if gen < max_gen:
                ratio = gen / max_gen
                new_pop_size = int(N_min + (N_init - N_min) * (1 - 1.5 * ratio**2))
                new_pop_size = max(N_min, min(pop_size, new_pop_size))
            else:
                new_pop_size = N_min
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                # Trim archive proportionally
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]
                # Update archive size
                archive_size = max(pop_size, int(1.5 * pop_size))

            # pbest rate increases with generations (more greedy later)
            p = 0.2 + 0.3 * (gen / max_gen) ** 2
            p = min(p, 0.4)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            # Generate mutation and crossover for each individual
            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Select r1, r2 (distinct from i) from pop+archive
                union = list(range(pop_size)) + list(range(len(archive)))
                valid = [j for j in union if j != i]
                if len(valid) >= 2:
                    r1, r2 = np.random.choice(valid, 2, replace=False)
                    def get_ind(idx):
                        return pop[idx] if idx < pop_size else archive[idx - pop_size]
                    x_r1 = get_ind(r1)
                    x_r2 = get_ind(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F, CR from memory with small perturbation
                r = np.random.randint(mem_size)
                F = np.clip(mem_F[r] + 0.05 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(mem_CR[r] + 0.1 * np.random.randn(), 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial (70%) or exponential (30%)
                trial = np.empty(dim)
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

                # Boundary handling: reflection
                for d in range(dim):
                    if trial[d] < lb[d]:
                        trial[d] = lb[d] + (lb[d] - trial[d]) % (ub[d] - lb[d])
                    elif trial[d] > ub[d]:
                        trial[d] = ub[d] - (trial[d] - ub[d]) % (ub[d] - lb[d])

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Update archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Remove randomly to keep diversity
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
                mem_F[mem_idx] = np.clip(F_lehmer, 0.1, 1.0)
                mem_CR[mem_idx] = np.clip(CR_mean, 0.0, 1.0)
                mem_idx = (mem_idx + 1) % mem_size

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 15:
                success_rates.pop(0)

            # ---------- Adaptive Local Search (L-BFGS) ----------
            # Trigger when: budget sufficient, diversity moderate, and recent success low
            if (gen % ls_freq == 0 and
                (self.budget - evals) > ls_min_budget and
                np.std(fitness) < 0.8 and
                (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.2)):

                h = 1e-5 * (ub - lb) + 1e-8
                # Forward difference gradient
                def grad_fwd(x):
                    fx = func(x)
                    g = np.zeros(dim)
                    for d in range(dim):
                        x_plus = np.clip(x + np.eye(1, dim, d)[0] * h[d], lb, ub)
                        g[d] = (func(x_plus) - fx) / h[d]
                    return g, fx

                x = self.x_opt.copy()
                f = self.f_opt
                for it in range(ls_max_iter):
                    if evals + dim + 2 >= self.budget:
                        break
                    g, fx = grad_fwd(x)
                    evals += dim + 1  # for gradient (including base evaluation)
                    if np.linalg.norm(g) < 1e-12:
                        break

                    # L-BFGS two-loop recursion
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        sy = np.dot(s_list[i], y_list[i]) + 1e-30
                        alpha[i] = np.dot(s_list[i], q) / sy
                        q = q - alpha[i] * y_list[i]
                    d = -q.copy()
                    if len(s_list) > 0:
                        last_sy = np.dot(s_list[-1], y_list[-1])
                        last_yy = np.dot(y_list[-1], y_list[-1]) + 1e-30
                        H0 = last_sy / last_yy
                        d = H0 * d
                    for i in range(len(s_list)):
                        sy = np.dot(s_list[i], y_list[i]) + 1e-30
                        beta = np.dot(y_list[i], d) / sy
                        d = d + (alpha[i] - beta) * s_list[i]

                    # Armijo line search
                    alpha_step = 1.0
                    c = 1e-4
                    for _ in range(12):
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if evals >= self.budget:
                            break
                        if f_new <= fx + c * alpha_step * np.dot(g, x_new - x):
                            break
                        alpha_step *= 0.5
                    if alpha_step < 1e-12:
                        break
                    # Update L-BFGS history (s, y)
                    s = x_new - x
                    y = grad_fwd(x_new)[0] - g  # recompute gradient at x_new
                    evals += dim + 1
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

                # Inject best found into population
                if self.f_opt < fitness.max() and evals < self.budget:
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # Add a locally perturbed point for diversity
                    perturbed = self.x_opt + 0.005 * np.random.randn(dim) * (ub - lb)
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

            # ---------- Stagnation-based Restart (soft) ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen)):
                n_restart = max(1, int(0.4 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Replace worst individuals with: half near best + half global LHS
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
                        # LHS-like point
                        perm = np.random.permutation(n_restart)
                        for j in range(dim):
                            pop[idx, j] = lb[j] + (perm[j] + 0.5) / n_restart * (ub[j] - lb[j])
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory partially
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                mem_idx = 0
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = min(max_gen // 3, ls_freq + 2)

        return self.f_opt, self.x_opt