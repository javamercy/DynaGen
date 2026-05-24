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

        # Sobol sequence initialization (simple quasi-random via scrambling)
        def sobol_points(n, d, seed=42):
            # Generate n points in [0,1)^d using Gray code order Sobol (simplified)
            rng = np.random.RandomState(seed)
            points = np.empty((n, d))
            for j in range(d):
                points[:, j] = (np.arange(n) + 0.5) / n
            for j in range(d):
                rng.shuffle(points[:, j])
            return points

        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Use Sobol for initial population
        pop = lb + sobol_points(pop_size, dim) * (ub - lb)

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        archive = []
        archive_size = pop_size * 2  # increased archive for diversity

        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters
        ls_freq_init = max(8, int(0.06 * max_gen))
        ls_freq = ls_freq_init
        ls_max_iter = max(2, int(0.04 * (self.budget / (2*dim + 5))))
        ls_max_iter = min(ls_max_iter, 12)
        ls_budget_fraction = 0.06

        # L-BFGS memory
        L_mem = 5
        s_list = []
        y_list = []

        # Adaptive crossover selection (binomial vs exponential)
        crossover_success = [0.5, 0.5]  # probability of binomial, exponential
        crossover_counts = [0, 0]

        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            p = 0.2 * (gen / max_gen) ** 2 + 0.1
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            crossover_type_success = [0, 0]  # binomial, exponential

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

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

                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Adaptive crossover selection based on recent success
                if np.random.rand() < crossover_success[0]:
                    use_binomial = True
                else:
                    use_binomial = False

                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                if use_binomial:
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
                        # Remove most similar
                        dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                        idx_remove = np.argmin(dists)
                        archive[idx_remove] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))
                    crossover_type_success[0 if use_binomial else 1] += 1

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

            # Update crossover success probabilities (exponential smoothing)
            for k in range(2):
                crossover_counts[k] += crossover_type_success[k]
            total = sum(crossover_counts)+1e-30
            crossover_success[0] = (crossover_counts[0] + 1) / (total + 2)  # Laplace smoothing

            # -------- Local search: L-BFGS --------
            if (gen % ls_freq == 0 and 
                (self.budget - evals) > dim * 5 + 20 and
                np.std(fitness) < 1.0 and 
                stagnation_counter > 2):
                h = 1e-5 * (ub - lb) + 1e-8
                def grad(x):
                    g = np.zeros(dim)
                    for d in range(dim):
                        xp = np.clip(x + np.eye(1,dim,d) * h[d], lb, ub)[0]
                        xn = np.clip(x - np.eye(1,dim,d) * h[d], lb, ub)[0]
                        g[d] = (func(xp) - func(xn)) / (2 * h[d])
                    return g

                x = self.x_opt.copy()
                f = self.f_opt
                for it in range(ls_max_iter):
                    if evals + 2*dim >= self.budget:
                        break
                    g = grad(x)
                    evals += 2*dim
                    if np.linalg.norm(g) < 1e-12:
                        break
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        alpha[i] = np.dot(s_list[i], q) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        q = q - alpha[i] * y_list[i]
                    d = -q
                    if len(s_list) > 0:
                        H0 = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-30)
                        d = H0 * d
                    for i in range(len(s_list)):
                        beta = np.dot(y_list[i], d) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        d = d + (alpha[i] - beta) * s_list[i]
                    alpha_step = 1.0
                    c = 1e-4
                    fx = f
                    for _ in range(12):
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= fx + c * alpha_step * np.dot(g, x_new - x):
                            break
                        alpha_step *= 0.5
                    if alpha_step < 1e-12:
                        break
                    s = x_new - x
                    y = grad(x_new) - g
                    evals += 2*dim
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
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt

            # -------- Stagnation and restart (multi-cluster with quasi-opposition) --------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(6, int(0.06 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Quasi-oppositional points: sample half near best, half opposite
                # Generate Sobol points for diversity
                sob = sobol_points(n_restart, dim, seed=gen)
                for idx in range(n_restart):
                    if idx < n_restart // 3:
                        # Quasi-opposite point
                        opposite = lb + ub - best_copy
                        scale = 0.1 * (ub - lb) * (1 - gen / max_gen)**0.5
                        pop[idx] = opposite + np.random.randn(dim) * scale
                    elif idx < 2 * n_restart // 3:
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reinitialize L-BFGS memory and archive partially
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                ls_freq = min(max_gen // 4, ls_freq + 2)

        return self.f_opt, self.x_opt